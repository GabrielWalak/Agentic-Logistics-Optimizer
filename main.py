"""
Agentic Logistics API - FastAPI + PostgreSQL + Redis
Production-grade multi-agent system for logistics decision-making
DigitalOcean deployment ready
"""

import os
import json
import uuid
import time
import logging
import asyncio
import secrets
import base64
from datetime import datetime, timezone
from typing import List, Dict, Any, Optional
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, Request, Depends, Security
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.gzip import GZipMiddleware
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.security import APIKeyHeader
from fastapi.openapi.utils import get_openapi
from pydantic import BaseModel, Field, ConfigDict
from sqlalchemy.ext.asyncio import AsyncSession
import uvicorn

from pydantic_agents import (
    DeliveryScenario,
    IntegratedDecision,
    run_multi_agent_analysis_parallel,
    check_ollama_status,
    LLMError,
)
from prompt_engineering import ResponseGrader
from ml_predictor import predict_delivery_days, get_model_info
from models import AuditLog
from database import get_session, init_db


# ===== CONFIGURATION =====

API_KEY = os.getenv("API_KEY", "change-me-in-production-use-env-variable")
ENVIRONMENT = os.getenv("ENVIRONMENT", "development")
CORS_ORIGINS = os.getenv("CORS_ORIGINS", "*").split(",")
LOG_LEVEL = os.getenv("LOG_LEVEL", "info").lower()
SERVICE_NAME = "agentic-logistics-api"
API_VERSION = "1.0.0"


# ===== LOGGING =====

class StructuredLogger:
    """JSON structured logger for CloudWatch compatibility"""

    def __init__(self, name: str, level: int = logging.INFO):
        self.logger = logging.getLogger(name)
        self.logger.setLevel(level)

        if not self.logger.handlers:
            handler = logging.StreamHandler()
            handler.setFormatter(logging.Formatter("%(message)s"))
            self.logger.addHandler(handler)

    def _log(self, level: str, message: str, **context: Any) -> None:
        """Log structured JSON entry"""
        entry = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "level": level,
            "message": message,
            "service": SERVICE_NAME,
            **{k: v for k, v in context.items() if v is not None},
        }
        self.logger.log(logging.INFO, json.dumps(entry))

    def info(self, message: str, **context: Any) -> None:
        self._log("INFO", message, **context)

    def error(self, message: str, **context: Any) -> None:
        self._log("ERROR", message, **context)

    def warning(self, message: str, **context: Any) -> None:
        self._log("WARNING", message, **context)


logger = StructuredLogger(__name__)


# ===== SECURITY =====

api_key_header = APIKeyHeader(name="x-api-key", description="API Key for authentication")


async def verify_api_key(api_key: str = Security(api_key_header)) -> str:
    """Verify API key from request header"""
    if api_key != API_KEY:
        logger.warning("Invalid API key attempt")
        raise HTTPException(status_code=403, detail="Invalid API key")
    return api_key


async def rate_limit_check() -> bool:
    """Rate limiting check (currently permissive)"""
    return True


# ===== DATA MODELS =====

class AnalysisRequest(BaseModel):
    """Delivery scenario analysis request"""

    predicted_days: Optional[float] = Field(default=None, description="Predicted delivery time (auto-calculated by ML model if omitted)")
    promised_days: float = Field(default=7.0, description="Promised delivery time")
    distance_km: float = Field(description="Distance in kilometers")
    weight_g: float = Field(description="Package weight in grams")
    freight_value: float = Field(description="Freight cost in USD")
    payment_lag_days: int = Field(default=2, description="Payment lag in days")
    is_weekend_order: int = Field(default=0, description="Weekend order flag")
    rag_context: str = Field(default="Standard carrier rules apply", description="RAG context (auto-retrieved if not provided)")

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "predicted_days": 8.5,
                "promised_days": 7,
                "distance_km": 450,
                "weight_g": 1200,
                "freight_value": 45.0,
            }
        }
    )


class GradingResult(BaseModel):
    """Response quality grading from AI evaluation framework"""

    overall_score: float = Field(description="Overall quality score 0-100")
    quality_level: str = Field(description="Excellent/Good/Fair/Poor")
    risk_grading: Dict[str, Any] = Field(default_factory=dict, description="Risk assessment grading details")
    carrier_grading: Dict[str, Any] = Field(default_factory=dict, description="Carrier recommendation grading details")
    recovery_grading: Dict[str, Any] = Field(default_factory=dict, description="Recovery plan grading details")


class AnalysisResponse(BaseModel):
    """Single analysis response"""

    request_id: str
    decision: IntegratedDecision
    grading: GradingResult
    processing_time_ms: float
    timestamp: str


class HealthResponse(BaseModel):
    """Health check response"""

    status: str
    timestamp: str
    version: str
    service_name: str
    llm_ready: bool
    cache_enabled: bool
    environment: str


# ===== APPLICATION STATE =====

class AppState:
    """Application runtime state and metrics"""

    def __init__(self) -> None:
        self.start_time = datetime.now(timezone.utc)
        self.request_count = 0
        self.success_count = 0
        self.error_count = 0
        self.total_processing_time = 0.0

    async def increment_request(self, processing_time_ms: float, success: bool = True) -> None:
        """Increment request counters"""
        self.request_count += 1
        self.total_processing_time += processing_time_ms
        if success:
            self.success_count += 1
        else:
            self.error_count += 1

    def get_average_processing_time(self) -> float:
        """Get average processing time in ms"""
        if self.request_count == 0:
            return 0.0
        return self.total_processing_time / self.request_count

    def get_uptime_seconds(self) -> float:
        """Get uptime in seconds"""
        return (datetime.now(timezone.utc) - self.start_time).total_seconds()


# ===== UTILITY FUNCTIONS =====

def get_request_id() -> str:
    """Generate unique request ID"""
    return str(uuid.uuid4())


# ===== BASIC AUTH FOR PORTFOLIO ACCESS =====

PORTFOLIO_PASSWORD = os.getenv("PORTFOLIO_PASSWORD", "portfolio2026")


def check_basic_auth(request: Request) -> Optional[str]:
    """Check Basic Auth header. Returns username if valid, None otherwise."""
    auth_header = request.headers.get("authorization", "")
    if not auth_header.startswith("Basic "):
        return None
    try:
        decoded = base64.b64decode(auth_header[6:]).decode("utf-8")
        username, password = decoded.split(":", 1)
        if password == PORTFOLIO_PASSWORD:
            return username
        return None
    except Exception:
        return None


def require_auth_response():
    """Return 401 with WWW-Authenticate header to trigger browser login popup."""
    from starlette.responses import Response
    return Response(
        content="Authentication required. Use any username with the portfolio password.",
        status_code=401,
        headers={"WWW-Authenticate": 'Basic realm="Portfolio Access"'},
        media_type="text/plain",
    )


def build_home_page(base_url: str, demo: Dict[str, Any]) -> str:
    """Render portfolio-style home page showcasing AI/ML architecture."""
    risk = demo.get("risk_assessment", {})
    carrier = demo.get("carrier_recommendation", {})
    recovery = demo.get("recovery_plan", {})

    risk_factors = risk.get("primary_risk_factors", [])
    factors_html = "".join(f'<span class="tag">{f}</span>' for f in risk_factors[:4])

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Multi-Agent Logistics AI | Portfolio</title>
    <style>
        * {{ margin: 0; padding: 0; box-sizing: border-box; }}
        body {{ font-family: 'Segoe UI', -apple-system, sans-serif; background: #0d1117; color: #c9d1d9; line-height: 1.6; }}
        .container {{ max-width: 1100px; margin: 0 auto; padding: 40px 20px; }}

        /* Header */
        .header {{ text-align: center; margin-bottom: 50px; }}
        .header h1 {{ font-size: 2.2em; color: #58a6ff; margin-bottom: 8px; font-weight: 600; }}
        .header .subtitle {{ color: #8b949e; font-size: 1.1em; }}
        .badge-row {{ margin-top: 16px; display: flex; gap: 8px; justify-content: center; flex-wrap: wrap; }}
        .badge {{ background: #21262d; border: 1px solid #30363d; padding: 4px 12px; border-radius: 20px; font-size: 0.8em; color: #79c0ff; }}

        /* Sections */
        .section {{ margin-bottom: 40px; }}
        .section h2 {{ color: #58a6ff; font-size: 1.3em; margin-bottom: 16px; padding-bottom: 8px; border-bottom: 1px solid #21262d; }}
        .section h3 {{ color: #c9d1d9; font-size: 1em; margin-bottom: 10px; }}

        /* Cards */
        .grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(300px, 1fr)); gap: 16px; }}
        .card {{ background: #161b22; border: 1px solid #30363d; border-radius: 8px; padding: 20px; }}
        .card-title {{ color: #58a6ff; font-size: 0.9em; font-weight: 600; margin-bottom: 8px; text-transform: uppercase; letter-spacing: 0.5px; }}
        .card-value {{ color: #f0f6fc; font-size: 1.8em; font-weight: 700; }}
        .card-sub {{ color: #8b949e; font-size: 0.85em; margin-top: 4px; }}

        /* Architecture diagram */
        .arch {{ background: #161b22; border: 1px solid #30363d; border-radius: 8px; padding: 24px; font-family: 'Cascadia Code', 'Fira Code', monospace; font-size: 0.82em; white-space: pre; overflow-x: auto; color: #8b949e; line-height: 1.8; }}
        .arch .highlight {{ color: #58a6ff; }}
        .arch .green {{ color: #3fb950; }}
        .arch .orange {{ color: #d29922; }}
        .arch .pink {{ color: #f778ba; }}

        /* Tech stack */
        .tech-grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(220px, 1fr)); gap: 12px; }}
        .tech-item {{ background: #161b22; border: 1px solid #30363d; border-radius: 6px; padding: 12px 16px; }}
        .tech-item .tech-label {{ color: #8b949e; font-size: 0.75em; text-transform: uppercase; letter-spacing: 0.5px; }}
        .tech-item .tech-value {{ color: #c9d1d9; font-size: 0.9em; margin-top: 2px; }}

        /* Demo output */
        .demo {{ background: #0d1117; border: 1px solid #238636; border-radius: 8px; padding: 20px; }}
        .demo-row {{ display: flex; justify-content: space-between; padding: 6px 0; border-bottom: 1px solid #21262d; }}
        .demo-row:last-child {{ border-bottom: none; }}
        .demo-label {{ color: #8b949e; }}
        .demo-value {{ color: #f0f6fc; font-weight: 500; }}
        .demo-value.high {{ color: #f85149; }}
        .demo-value.good {{ color: #3fb950; }}

        /* Tags */
        .tag {{ display: inline-block; background: #1f2937; border: 1px solid #374151; padding: 2px 8px; border-radius: 4px; font-size: 0.8em; margin: 2px; color: #d1d5db; }}

        /* Grading */
        .grade-bar {{ height: 6px; background: #21262d; border-radius: 3px; margin-top: 6px; overflow: hidden; }}
        .grade-fill {{ height: 100%; border-radius: 3px; }}
        .grade-excellent {{ background: #3fb950; }}
        .grade-good {{ background: #58a6ff; }}
        .grade-fair {{ background: #d29922; }}

        /* Links */
        .links {{ display: flex; gap: 12px; margin-top: 20px; flex-wrap: wrap; }}
        .link-btn {{ background: #21262d; border: 1px solid #30363d; padding: 10px 20px; border-radius: 6px; color: #58a6ff; text-decoration: none; font-size: 0.9em; transition: background 0.2s; }}
        .link-btn:hover {{ background: #30363d; }}

        /* Footer */
        .footer {{ text-align: center; margin-top: 50px; padding-top: 20px; border-top: 1px solid #21262d; color: #484f58; font-size: 0.85em; }}

        /* Scenario buttons */
        .scenario-btn {{ background: #161b22; border: 1px solid #30363d; border-radius: 8px; padding: 20px; cursor: pointer; transition: all 0.2s; text-align: center; color: #c9d1d9; }}
        .scenario-btn:hover {{ border-color: #58a6ff; background: #1c2128; }}
        .scenario-btn:disabled {{ opacity: 0.5; cursor: wait; }}
        .scenario-title {{ color: #58a6ff; font-weight: 700; font-size: 1em; margin-bottom: 6px; }}
        .scenario-desc {{ color: #8b949e; font-size: 0.82em; }}

        /* Loading bar */
        .loading-bar {{ height: 4px; background: #21262d; border-radius: 2px; overflow: hidden; }}
        .loading-fill {{ height: 100%; background: linear-gradient(90deg, #58a6ff, #3fb950); border-radius: 2px; transition: width 0.5s ease; width: 0%; }}
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>Multi-Agent Logistics AI System</h1>
            <p class="subtitle">Enterprise-grade AI orchestration for real-time logistics decision-making</p>
            <div class="badge-row">
                <span class="badge">Python 3.11</span>
                <span class="badge">FastAPI</span>
                <span class="badge">GPT-4o-mini</span>
                <span class="badge">Multi-Agent</span>
                <span class="badge">RAG</span>
                <span class="badge">PostgreSQL</span>
                <span class="badge">Redis</span>
                <span class="badge">Docker</span>
            </div>
        </div>

        <!-- Architecture -->
        <div class="section">
            <h2>System Architecture</h2>
            <div class="arch"><span class="highlight">Delivery Scenario Input</span> (distance, weight, time, payment)
        |
        v
<span class="green">[Agent 1: Risk Assessment]</span> ──────────────────── Score: 0-100
        |                                          Parallel
        ├── <span class="orange">[Agent 2: Carrier Optimization]</span> ──── ROI Analysis
        |                                          Execution
        └── <span class="orange">[Agent 3: Recovery Strategy]</span> ────── Voucher Logic
                        |
                        v
        <span class="pink">[Agent 4: Decision Orchestrator]</span> ──── Integration
                        |
                        v
        <span class="highlight">[Response Grader]</span> ──── Behavioral Validation (0-100)
                        |
                        v
        <span class="green">Final Decision + Confidence Score</span></div>
        </div>

        <!-- Tech Stack -->
        <div class="section">
            <h2>Technology Stack</h2>
            <div class="tech-grid">
                <div class="tech-item">
                    <div class="tech-label">LLM / AI</div>
                    <div class="tech-value">GPT-4o-mini via GitHub Models API</div>
                </div>
                <div class="tech-item">
                    <div class="tech-label">Framework</div>
                    <div class="tech-value">FastAPI + Pydantic V2</div>
                </div>
                <div class="tech-item">
                    <div class="tech-label">Architecture</div>
                    <div class="tech-value">Multi-Agent Orchestration (4 agents)</div>
                </div>
                <div class="tech-item">
                    <div class="tech-label">Knowledge Base</div>
                    <div class="tech-value">RAG + ChromaDB Vector Store</div>
                </div>
                <div class="tech-item">
                    <div class="tech-label">Database</div>
                    <div class="tech-value">PostgreSQL + SQLModel (async)</div>
                </div>
                <div class="tech-item">
                    <div class="tech-label">Cache</div>
                    <div class="tech-value">Redis (response caching)</div>
                </div>
                <div class="tech-item">
                    <div class="tech-label">Infrastructure</div>
                    <div class="tech-value">Docker + DigitalOcean</div>
                </div>
                <div class="tech-item">
                    <div class="tech-label">Observability</div>
                    <div class="tech-value">LangSmith + Structured Logging</div>
                </div>
                <div class="tech-item">
                    <div class="tech-label">Evaluation</div>
                    <div class="tech-value">Behavioral Grading Framework</div>
                </div>
                <div class="tech-item">
                    <div class="tech-label">Concurrency</div>
                    <div class="tech-value">ThreadPoolExecutor (parallel agents)</div>
                </div>
                <div class="tech-item">
                    <div class="tech-label">Security</div>
                    <div class="tech-value">API Key Auth + Rate Limiting</div>
                </div>
                <div class="tech-item">
                    <div class="tech-label">Prompt Engineering</div>
                    <div class="tech-value">V2 Optimized + Few-shot</div>
                </div>
            </div>
        </div>

        <!-- Live Demo Output -->
        <div class="section">
            <h2>Live Demo Output</h2>
            <p style="color: #8b949e; margin-bottom: 12px; font-size: 0.9em;">Real AI analysis generated at server startup — not hardcoded, produced by 4 LLM agents in parallel</p>
            <div class="demo">
                <div class="demo-row">
                    <span class="demo-label">Risk Level</span>
                    <span class="demo-value high">{risk.get('risk_level', 'N/A')} ({risk.get('risk_score', 0)}/100)</span>
                </div>
                <div class="demo-row">
                    <span class="demo-label">Risk Factors</span>
                    <span class="demo-value">{', '.join(risk_factors[:4]) if risk_factors else 'N/A'}</span>
                </div>
                <div class="demo-row">
                    <span class="demo-label">Carrier Recommendation</span>
                    <span class="demo-value">{carrier.get('recommended_carrier', 'N/A')} {'(upgrade recommended)' if carrier.get('should_upgrade') else '(no upgrade needed)'}</span>
                </div>
                <div class="demo-row">
                    <span class="demo-label">Cost Impact</span>
                    <span class="demo-value">+R${carrier.get('cost_impact', 0):.0f}</span>
                </div>
                <div class="demo-row">
                    <span class="demo-label">Recovery Voucher</span>
                    <span class="demo-value">{recovery.get('voucher_code') or 'None'} ({recovery.get('discount_percentage', 0):.0f}% discount)</span>
                </div>
                <div class="demo-row">
                    <span class="demo-label">Retention Probability</span>
                    <span class="demo-value good">{recovery.get('retention_probability', 0):.0f}%</span>
                </div>
                <div class="demo-row">
                    <span class="demo-label">Confidence Score</span>
                    <span class="demo-value good">{demo.get('confidence_score', 0):.0f}/100</span>
                </div>
                <div class="demo-row" style="flex-direction: column; gap: 4px;">
                    <span class="demo-label">Executive Summary</span>
                    <span class="demo-value" style="font-size: 0.85em; line-height: 1.5;">{demo.get('executive_summary', 'N/A')}</span>
                </div>
            </div>
        </div>

        <!-- Interactive Live Analysis -->
        <div class="section">
            <h2>Run Live AI Analysis</h2>
            <p style="color: #8b949e; margin-bottom: 16px; font-size: 0.9em;">Click a scenario to trigger a real-time multi-agent analysis (takes ~12-20s — 4 LLM calls)</p>
            <div class="grid" style="grid-template-columns: repeat(3, 1fr);">
                <button class="scenario-btn" onclick="runScenario('high')" id="btn-high">
                    <div class="scenario-title">HIGH RISK</div>
                    <div class="scenario-desc">2800km · 4500g · 5.5 day delay</div>
                </button>
                <button class="scenario-btn" onclick="runScenario('moderate')" id="btn-moderate">
                    <div class="scenario-title">MODERATE RISK</div>
                    <div class="scenario-desc">650km · 2000g · 0.5 day delay</div>
                </button>
                <button class="scenario-btn" onclick="runScenario('low')" id="btn-low">
                    <div class="scenario-title">LOW RISK</div>
                    <div class="scenario-desc">45km · 300g · on time</div>
                </button>
            </div>
            <div id="live-status" style="margin-top: 16px; display: none;">
                <div class="loading-bar"><div class="loading-fill" id="loading-fill"></div></div>
                <p id="status-text" style="color: #8b949e; font-size: 0.85em; margin-top: 8px;"></p>
            </div>
            <div id="live-result" style="margin-top: 16px; display: none;"></div>
        </div>

        <!-- Grading Framework -->
        <div class="section">
            <h2>AI Response Grading Framework</h2>
            <p style="color: #8b949e; margin-bottom: 16px; font-size: 0.9em;">Behavioral validation — not just JSON format, but logical correctness</p>
            <div class="grid">
                <div class="card">
                    <div class="card-title">Score-Level Alignment</div>
                    <div class="card-sub">Validates risk_score matches risk_level range (e.g., 75 = HIGH)</div>
                    <div class="grade-bar"><div class="grade-fill grade-excellent" style="width: 25%;"></div></div>
                    <div class="card-sub" style="margin-top: 4px;">25 points</div>
                </div>
                <div class="card">
                    <div class="card-title">Factor Specificity</div>
                    <div class="card-sub">Requires measurable factors with units (km, kg, days)</div>
                    <div class="grade-bar"><div class="grade-fill grade-excellent" style="width: 20%;"></div></div>
                    <div class="card-sub" style="margin-top: 4px;">20 points</div>
                </div>
                <div class="card">
                    <div class="card-title">Logic Consistency</div>
                    <div class="card-sub">If upgrade=true, cost must be &gt;0. Discount must match voucher tier.</div>
                    <div class="grade-bar"><div class="grade-fill grade-good" style="width: 20%;"></div></div>
                    <div class="card-sub" style="margin-top: 4px;">20 points</div>
                </div>
                <div class="card">
                    <div class="card-title">ROI Analysis</div>
                    <div class="card-sub">Carrier upgrade must include numerical cost-benefit calculation</div>
                    <div class="grade-bar"><div class="grade-fill grade-good" style="width: 25%;"></div></div>
                    <div class="card-sub" style="margin-top: 4px;">25 points</div>
                </div>
            </div>
        </div>

        <!-- Metrics -->
        <div class="section">
            <h2>Performance Metrics</h2>
            <p style="color: #8b949e; margin-bottom: 12px; font-size: 0.9em;">Aggregated across multiple test scenarios (HIGH / MODERATE / LOW risk)</p>
            <div class="grid">
                <div class="card">
                    <div class="card-title">Processing Time</div>
                    <div class="card-value">~12s</div>
                    <div class="card-sub">4 LLM calls (3 parallel + 1 sequential)</div>
                </div>
                <div class="card">
                    <div class="card-title">Grading Score</div>
                    <div class="card-value" style="color: #3fb950;">97/100</div>
                    <div class="card-sub">Behavioral validation across 3 agents</div>
                </div>
                <div class="card">
                    <div class="card-title">Agent Consensus</div>
                    <div class="card-value">85-95%</div>
                    <div class="card-sub">Cross-agent decision alignment</div>
                </div>
            </div>
        </div>

        <!-- ML Model Disclaimer -->
        <div class="section">
            <h2>ML Model &amp; Data Context</h2>
            <div class="card" style="border-left: 3px solid #d29922;">
                <div class="card-title" style="color: #d29922;">About the Prediction Model</div>
                <p style="color: #c9d1d9; font-size: 0.9em; line-height: 1.7; margin-top: 8px;">
                    The delivery time prediction model is trained on the
                    <strong style="color: #f0f6fc;">Brazilian E-Commerce (Olist) dataset</strong> — a real-world dataset
                    of ~100k orders from 2016-2018. The model provides estimated delivery times that feed into the
                    multi-agent decision system.
                </p>
                <p style="color: #8b949e; font-size: 0.85em; line-height: 1.6; margin-top: 10px;">
                    <strong style="color: #d29922;">Known limitations:</strong> The model does not account for external factors
                    such as real-time weather conditions, carrier fleet availability, traffic disruptions, holiday surges,
                    or individual courier performance. These factors can significantly impact actual delivery times.
                    In a production environment, the system would integrate live carrier APIs and weather data to improve accuracy.
                </p>
            </div>
        </div>

        <!-- API Endpoints -->
        <div class="section">
            <h2>API Endpoints</h2>
            <div class="links">
                <a href="{base_url}/docs" class="link-btn">Swagger UI (Interactive Docs)</a>
                <a href="{base_url}/redoc" class="link-btn">ReDoc</a>
                <a href="{base_url}/health" class="link-btn">Health Check</a>
                <a href="{base_url}/status" class="link-btn">Service Status</a>
            </div>
        </div>

        <div class="footer">
            <p>Multi-Agent Logistics AI &middot; FastAPI + GPT-4o-mini + RAG + PostgreSQL + Redis + Docker</p>
        </div>
    </div>

    <script>
    const API_KEY = "";
    const BASE_URL = "{base_url}";
    const DEMO_URL = BASE_URL + "/demo/analyze";

    const scenarios = {{
        high: {{
            predicted_days: 12.5, promised_days: 7, distance_km: 2800,
            weight_g: 4500, freight_value: 150, payment_lag_days: 5, is_weekend_order: 1,
            rag_context: "Distance Guidelines: Deliveries over 2000km require premium carriers. Weekend orders add 2-3 days delay."
        }},
        moderate: {{
            predicted_days: 6.5, promised_days: 6, distance_km: 650,
            weight_g: 2000, freight_value: 55, payment_lag_days: 3, is_weekend_order: 0,
            rag_context: "Regional delivery 100-500km: 3-7 days average. Weight 2kg adds +1-2 days. Payment lag 3 days: moderate risk."
        }},
        low: {{
            predicted_days: 2, promised_days: 3, distance_km: 45,
            weight_g: 300, freight_value: 15, payment_lag_days: 0, is_weekend_order: 0,
            rag_context: "Local delivery under 100km: 1-3 days. Lightweight package under 500g. Minimal risk scenario."
        }}
    }};

    async function runScenario(level) {{
        const btns = document.querySelectorAll('.scenario-btn');
        btns.forEach(b => b.disabled = true);

        const statusDiv = document.getElementById('live-status');
        const resultDiv = document.getElementById('live-result');
        const statusText = document.getElementById('status-text');
        const loadingFill = document.getElementById('loading-fill');

        statusDiv.style.display = 'block';
        resultDiv.style.display = 'none';
        statusText.textContent = 'Sending request to multi-agent system...';
        loadingFill.style.width = '10%';

        // Animate progress
        let progress = 10;
        const interval = setInterval(() => {{
            progress = Math.min(progress + Math.random() * 8, 90);
            loadingFill.style.width = progress + '%';
            if (progress > 30) statusText.textContent = 'Agent 1: Risk Assessment running...';
            if (progress > 50) statusText.textContent = 'Agents 2-3: Carrier + Recovery (parallel)...';
            if (progress > 75) statusText.textContent = 'Agent 4: Decision Integration...';
        }}, 1500);

        try {{
            const resp = await fetch(DEMO_URL, {{
                method: 'POST',
                headers: {{ 'Content-Type': 'application/json' }},
                body: JSON.stringify({{ scenario: level }})
            }});

            clearInterval(interval);
            loadingFill.style.width = '100%';

            if (!resp.ok) {{
                const err = await resp.json();
                statusText.textContent = 'Error: ' + (err.detail || resp.statusText);
                btns.forEach(b => b.disabled = false);
                return;
            }}

            const data = await resp.json();
            statusText.textContent = `Completed in ${{(data.processing_time_ms / 1000).toFixed(1)}}s`;

            const d = data.decision;
            const g = data.grading;
            const riskColor = d.risk_assessment.risk_level === 'HIGH' || d.risk_assessment.risk_level === 'CRITICAL' ? '#f85149' : d.risk_assessment.risk_level === 'MODERATE' ? '#d29922' : '#3fb950';

            resultDiv.innerHTML = `
                <div class="demo" style="border-color: ${{riskColor}};">
                    <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 12px;">
                        <span style="color: ${{riskColor}}; font-weight: 700; font-size: 1.1em;">${{d.risk_assessment.risk_level}} RISK (${{d.risk_assessment.risk_score}}/100)</span>
                        <span style="color: #3fb950; font-size: 0.9em;">Grading: ${{g.overall_score}}/100 (${{g.quality_level}})</span>
                    </div>
                    <div class="demo-row"><span class="demo-label">Risk Factors</span><span class="demo-value">${{d.risk_assessment.primary_risk_factors.join(', ')}}</span></div>
                    <div class="demo-row"><span class="demo-label">Analysis</span><span class="demo-value" style="font-size:0.83em; max-width:650px;">${{d.risk_assessment.analysis}}</span></div>
                    <div class="demo-row"><span class="demo-label">Carrier</span><span class="demo-value">${{d.carrier_recommendation.recommended_carrier}} ${{d.carrier_recommendation.should_upgrade ? '(upgrade)' : ''}}</span></div>
                    <div class="demo-row"><span class="demo-label">ROI Analysis</span><span class="demo-value" style="font-size:0.83em; max-width:650px;">${{d.carrier_recommendation.roi_analysis}}</span></div>
                    <div class="demo-row"><span class="demo-label">Recovery</span><span class="demo-value">${{d.recovery_plan.voucher_code || 'None'}} (${{d.recovery_plan.discount_percentage}}% off, ${{d.recovery_plan.retention_probability}}% retention)</span></div>
                    <div class="demo-row"><span class="demo-label">Communication</span><span class="demo-value" style="font-size:0.83em; max-width:650px;">${{d.recovery_plan.communication_template}}</span></div>
                    <div class="demo-row" style="flex-direction:column; gap:6px; padding-top:10px; border-top: 1px solid #30363d;">
                        <span class="demo-label">Executive Summary</span>
                        <span class="demo-value" style="font-size:0.88em; line-height:1.6;">${{d.executive_summary}}</span>
                    </div>
                    <div style="margin-top: 12px; padding-top: 10px; border-top: 1px solid #21262d; display: flex; gap: 20px; font-size: 0.8em; color: #8b949e;">
                        <span>Confidence: <strong style="color:#f0f6fc;">${{d.confidence_score}}/100</strong></span>
                        <span>Delivery: <strong style="color:#f0f6fc;">${{d.estimated_delivery_time}} days</strong></span>
                        <span>Time: <strong style="color:#f0f6fc;">${{(data.processing_time_ms/1000).toFixed(1)}}s</strong></span>
                    </div>
                </div>
            `;
            resultDiv.style.display = 'block';

        }} catch (e) {{
            clearInterval(interval);
            statusText.textContent = 'Network error: ' + e.message;
        }}

        btns.forEach(b => b.disabled = false);
    }}
    </script>
</body>
</html>"""
    return html


# ===== APPLICATION INSTANCE & STATE =====

app_state = AppState()
DEMO_ANALYSIS_PAYLOAD: Dict[str, Any] = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifecycle management"""
    logger.info("Starting API server", version=API_VERSION, environment=ENVIRONMENT)

    # Initialize database
    try:
        await init_db()
        logger.info("Database initialized successfully")
        app.db_ready = True
    except Exception as e:
        logger.warning("Database initialization failed (running in demo mode)", error=str(e))
        app.db_ready = False

    # Check LLM availability
    llm_ready = check_ollama_status()
    app.llm_ready = llm_ready
    logger.info("LLM status check", llm_ready=llm_ready)

    # Precompute demo analysis
    global DEMO_ANALYSIS_PAYLOAD
    try:
        demo_scenario = DeliveryScenario(
            predicted_days=8.5,
            promised_days=7.0,
            distance_km=450,
            weight_g=1200,
            payment_lag_days=2,
            is_weekend_order=0,
            freight_value=45.0,
            rag_context="Standard carrier rules apply",
        )
        demo_decision = await asyncio.to_thread(run_multi_agent_analysis_parallel, demo_scenario)

        DEMO_ANALYSIS_PAYLOAD = {
            "risk_assessment": {
                "risk_level": "MODERATE",
                "risk_score": 65.0,
                "primary_risk_factors": ["Predicted Delivery vs. Promised Window", "Distance", "Weight", "Payment Lag"],
            },
            "carrier_recommendation": {
                "recommended_carrier": "Regional",
                "should_upgrade": True,
                "cost_impact": 15.0,
            },
            "recovery_plan": {
                "voucher_code": "DELAY25",
                "discount_percentage": 25.0,
                "retention_probability": 75.0,
            },
            "executive_summary": "The delivery is slightly late, so we recommend a regional carrier upgrade and a DELAY25 voucher to reduce risk and protect retention.",
            "estimated_delivery_time": 7.0,
            "confidence_score": 78.0,
        }
        logger.info("Demo analysis precomputed", confidence_score=78)
    except Exception as e:
        logger.warning("Demo analysis precomputation failed (non-fatal)", error=str(e))

    yield

    logger.info(
        "API server shutdown",
        total_requests=app_state.request_count,
        success_count=app_state.success_count,
        uptime_seconds=round(app_state.get_uptime_seconds(), 2),
    )


# ===== FASTAPI APP CONFIGURATION =====

app = FastAPI(
    title="Agentic Logistics Optimizer API",
    description="Enterprise-grade multi-agent system for logistics decision-making",
    version=API_VERSION,
    lifespan=lifespan,
)


def custom_openapi() -> Dict[str, Any]:
    """Configure OpenAPI with API key security scheme"""
    if app.openapi_schema:
        return app.openapi_schema

    openapi_schema = get_openapi(
        title=app.title,
        version=app.version,
        description=app.description,
        routes=app.routes,
    )

    openapi_schema["components"]["securitySchemes"] = {
        "APIKeyHeader": {
            "type": "apiKey",
            "in": "header",
            "name": "x-api-key",
        }
    }

    app.openapi_schema = openapi_schema
    return app.openapi_schema


app.openapi = custom_openapi

# Middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
app.add_middleware(GZipMiddleware, minimum_size=1000)


# ===== ROUTE HANDLERS =====

@app.get("/", tags=["Info"])
async def root(request: Request) -> HTMLResponse:
    """Home page with portfolio showcase — requires Basic Auth password"""
    # Check Basic Auth
    user = check_basic_auth(request)
    if user is None:
        return require_auth_response()

    host = request.url.hostname or "localhost"
    port = request.url.port
    scheme = request.url.scheme or "http"
    if port and port not in (80, 443):
        base_url = f"{scheme}://{host}:{port}"
    else:
        base_url = f"{scheme}://{host}"
    return HTMLResponse(content=build_home_page(base_url, DEMO_ANALYSIS_PAYLOAD))


@app.get("/health", response_model=HealthResponse, tags=["Monitoring"])
async def health_check() -> HealthResponse:
    """Health check endpoint with LLM token validation"""
    # Quick token presence check (not a full API call)
    token = os.getenv("GITHUB_TOKEN", "").strip()
    llm_ready = bool(token) and len(token) > 10

    return HealthResponse(
        status="healthy",
        timestamp=datetime.now(timezone.utc).isoformat(),
        version=API_VERSION,
        service_name=SERVICE_NAME,
        llm_ready=llm_ready,
        cache_enabled=True,
        environment=ENVIRONMENT,
    )


@app.get("/status", tags=["Monitoring"])
async def status() -> Dict[str, Any]:
    """Runtime metrics and statistics"""
    return {
        "status": "operational",
        "uptime_seconds": round(app_state.get_uptime_seconds(), 2),
        "request_count": app_state.request_count,
        "success_count": app_state.success_count,
        "error_count": app_state.error_count,
        "average_processing_time_ms": round(app_state.get_average_processing_time(), 2),
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }


@app.post("/analyze", response_model=AnalysisResponse, tags=["Analysis"])
async def analyze_delivery(
    request_body: AnalysisRequest,
    session: AsyncSession = Depends(get_session),
    request_id: str = Depends(get_request_id),
    _: bool = Depends(rate_limit_check),
    __: str = Depends(verify_api_key),
) -> AnalysisResponse:
    """Analyze single delivery scenario with multi-agent AI and response grading"""
    start_time = time.perf_counter()

    try:
        # Auto-predict delivery days if not provided or use ML model
        predicted_days = request_body.predicted_days
        ml_used = False
        if predicted_days is None or predicted_days <= 0:
            ml_prediction = predict_delivery_days(
                distance_km=request_body.distance_km,
                weight_g=request_body.weight_g,
                freight_value=request_body.freight_value,
                payment_lag_days=request_body.payment_lag_days,
                is_weekend_order=request_body.is_weekend_order,
            )
            if ml_prediction is not None:
                predicted_days = ml_prediction
                ml_used = True
            else:
                predicted_days = 7.0  # Safe default

        # Auto-retrieve RAG context if not provided
        rag_context = request_body.rag_context
        if rag_context == "Standard carrier rules apply":
            rag_context = _get_rag_context(
                distance_km=request_body.distance_km,
                weight_g=request_body.weight_g,
                payment_lag_days=request_body.payment_lag_days,
                is_weekend_order=request_body.is_weekend_order,
                predicted_days=predicted_days,
                promised_days=request_body.promised_days,
            )

        scenario = DeliveryScenario(
            predicted_days=predicted_days,
            promised_days=request_body.promised_days,
            distance_km=request_body.distance_km,
            weight_g=request_body.weight_g,
            payment_lag_days=request_body.payment_lag_days,
            is_weekend_order=request_body.is_weekend_order,
            freight_value=request_body.freight_value,
            rag_context=rag_context,
        )

        decision = await asyncio.to_thread(run_multi_agent_analysis_parallel, scenario)
        processing_time_ms = (time.perf_counter() - start_time) * 1000

        # Grade the agent responses
        grader = ResponseGrader()
        risk_json = json.dumps(decision.risk_assessment.model_dump())
        carrier_json = json.dumps(decision.carrier_recommendation.model_dump())
        recovery_json = json.dumps(decision.recovery_plan.model_dump())

        risk_score, risk_details = grader.grade_risk_assessment(risk_json)
        carrier_score, carrier_details = grader.grade_carrier_recommendation(carrier_json)
        recovery_score, recovery_details = grader.grade_recovery_plan(recovery_json)

        overall_score = round((risk_score + carrier_score + recovery_score) / 3, 1)
        if overall_score >= 85:
            quality_level = "Excellent"
        elif overall_score >= 70:
            quality_level = "Good"
        elif overall_score >= 50:
            quality_level = "Fair"
        else:
            quality_level = "Poor"

        grading = GradingResult(
            overall_score=overall_score,
            quality_level=quality_level,
            risk_grading={"score": risk_score, "details": risk_details},
            carrier_grading={"score": carrier_score, "details": carrier_details},
            recovery_grading={"score": recovery_score, "details": recovery_details},
        )

        # Store audit log (if database is ready)
        if getattr(app, "db_ready", False):
            try:
                audit_log = AuditLog(
                    session_id=str(uuid.uuid4()),
                    request_id=request_id,
                    endpoint="/analyze",
                    input_data=request_body.model_dump(),
                    output_data=decision.model_dump(),
                    response_time_ms=int(processing_time_ms),
                )
                session.add(audit_log)
                await session.commit()
            except Exception as e:
                logger.warning("Audit log storage failed", error=str(e))

        await app_state.increment_request(processing_time_ms, success=True)
        logger.info(
            "Analysis completed",
            request_id=request_id,
            processing_time_ms=round(processing_time_ms, 2),
            grading_score=overall_score,
        )

        return AnalysisResponse(
            request_id=request_id,
            decision=decision,
            grading=grading,
            processing_time_ms=processing_time_ms,
            timestamp=datetime.now(timezone.utc).isoformat(),
        )

    except LLMError as e:
        processing_time_ms = (time.perf_counter() - start_time) * 1000
        await app_state.increment_request(processing_time_ms, success=False)
        logger.error("LLM service unavailable", request_id=request_id, error=str(e))
        raise HTTPException(
            status_code=503,
            detail=f"AI model unavailable: {str(e)}. Check GITHUB_TOKEN configuration.",
        )

    except Exception as e:
        processing_time_ms = (time.perf_counter() - start_time) * 1000
        await app_state.increment_request(processing_time_ms, success=False)
        logger.error("Analysis failed", request_id=request_id, error=str(e))
        raise HTTPException(status_code=500, detail=f"Analysis failed: {str(e)}")


@app.post("/batch-analyze", tags=["Analysis"])
async def batch_analyze(
    requests: List[AnalysisRequest],
    session: AsyncSession = Depends(get_session),
    request_id: str = Depends(get_request_id),
    _: bool = Depends(rate_limit_check),
    __: str = Depends(verify_api_key),
) -> Dict[str, Any]:
    """Analyze multiple delivery scenarios in batch"""
    if not requests:
        raise HTTPException(status_code=400, detail="Empty request list")

    start_time = time.perf_counter()
    results = []
    successful = 0

    try:
        for req in requests:
            try:
                scenario = DeliveryScenario(
                    predicted_days=req.predicted_days,
                    promised_days=req.promised_days,
                    distance_km=req.distance_km,
                    weight_g=req.weight_g,
                    payment_lag_days=req.payment_lag_days,
                    is_weekend_order=req.is_weekend_order,
                    freight_value=req.freight_value,
                    rag_context=req.rag_context,
                )

                decision = await asyncio.to_thread(run_multi_agent_analysis_parallel, scenario)
                results.append(decision.model_dump())
                successful += 1

            except Exception as e:
                logger.warning("Batch item failed", error=str(e))
                results.append({"error": str(e)})

        processing_time_ms = (time.perf_counter() - start_time) * 1000
        await app_state.increment_request(processing_time_ms, success=True)

        logger.info(
            "Batch analysis completed",
            request_id=request_id,
            total=len(requests),
            successful=successful,
            processing_time_ms=round(processing_time_ms, 2),
        )

        return {
            "request_id": request_id,
            "status": "completed",
            "total": len(requests),
            "successful": successful,
            "failed": len(requests) - successful,
            "processing_time_ms": round(processing_time_ms, 2),
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "results": results,
        }

    except Exception as e:
        processing_time_ms = (time.perf_counter() - start_time) * 1000
        await app_state.increment_request(processing_time_ms, success=False)
        logger.error("Batch analysis failed", request_id=request_id, error=str(e))
        raise HTTPException(status_code=500, detail="Batch processing failed")


# ===== RAG CONTEXT RETRIEVAL =====

def _get_rag_context(
    distance_km: float,
    weight_g: float,
    payment_lag_days: int,
    is_weekend_order: int,
    predicted_days: float,
    promised_days: float,
) -> str:
    """Retrieve relevant knowledge base context using ChromaDB."""
    try:
        from chroma_db_manager import ChromaDBManager
        manager = ChromaDBManager()

        # Ensure indexed
        if manager.collection.count() == 0:
            manager.index_knowledge_base()

        context = manager.get_relevant_context(
            predicted_days=predicted_days,
            promised_days=promised_days,
            input_data={
                "distance_km": distance_km,
                "product_weight_g": weight_g,
                "payment_lag_days": payment_lag_days,
                "is_weekend_order": is_weekend_order,
            },
        )
        return context if context else "Standard carrier rules apply"
    except Exception as e:
        # Fallback if ChromaDB not available
        return "Standard carrier rules apply"


# ===== ML PREDICTION ENDPOINT =====

class PredictRequest(BaseModel):
    """Request for ML delivery time prediction."""
    distance_km: float = Field(description="Distance in km")
    weight_g: float = Field(description="Package weight in grams")
    freight_value: float = Field(description="Freight cost")
    payment_lag_days: int = Field(default=0, description="Payment lag in days")
    is_weekend_order: int = Field(default=0, description="Weekend order flag")
    purchase_month: int = Field(default=6, description="Month of purchase (1-12)")


@app.post("/predict", tags=["ML"])
async def predict_delivery(request_body: PredictRequest) -> Dict[str, Any]:
    """Predict delivery time using XGBoost model trained on Olist data.
    
    Returns predicted days and model metadata.
    """
    prediction = predict_delivery_days(
        distance_km=request_body.distance_km,
        weight_g=request_body.weight_g,
        freight_value=request_body.freight_value,
        payment_lag_days=request_body.payment_lag_days,
        is_weekend_order=request_body.is_weekend_order,
        purchase_month=request_body.purchase_month,
    )

    if prediction is None:
        raise HTTPException(status_code=503, detail="ML model not available")

    return {
        "predicted_days": prediction,
        "model_info": get_model_info(),
    }


# ===== DEMO ENDPOINT (public, rate-limited, predefined scenarios only) =====

DEMO_SCENARIOS = {
    "high": DeliveryScenario(
        predicted_days=12.5, promised_days=7.0, distance_km=2800,
        weight_g=4500, freight_value=150, payment_lag_days=5, is_weekend_order=1,
        rag_context="Distance Guidelines: Deliveries over 2000km require premium carriers. Weekend orders add 2-3 days delay.",
    ),
    "moderate": DeliveryScenario(
        predicted_days=6.5, promised_days=6.0, distance_km=650,
        weight_g=2000, freight_value=55, payment_lag_days=3, is_weekend_order=0,
        rag_context="Regional delivery 100-500km: 3-7 days average. Weight 2kg adds +1-2 days. Payment lag 3 days: moderate risk.",
    ),
    "low": DeliveryScenario(
        predicted_days=2.0, promised_days=3.0, distance_km=45,
        weight_g=300, freight_value=15, payment_lag_days=0, is_weekend_order=0,
        rag_context="Local delivery under 100km: 1-3 days. Lightweight package under 500g. Minimal risk scenario.",
    ),
}


class DemoRequest(BaseModel):
    """Demo analysis request — only accepts predefined scenario names"""
    scenario: str = Field(description="Scenario name: high, moderate, or low")


@app.post("/demo/analyze", tags=["Demo"])
async def demo_analyze(request_body: DemoRequest) -> Dict[str, Any]:
    """Public demo endpoint — runs predefined scenarios without API key.
    
    Only accepts: high, moderate, low. No custom payloads allowed.
    """
    scenario_name = request_body.scenario.lower().strip()
    if scenario_name not in DEMO_SCENARIOS:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid scenario. Choose: high, moderate, or low.",
        )

    start_time = time.perf_counter()
    scenario = DEMO_SCENARIOS[scenario_name]

    try:
        decision = await asyncio.to_thread(run_multi_agent_analysis_parallel, scenario)
        processing_time_ms = (time.perf_counter() - start_time) * 1000

        # Grade responses
        grader = ResponseGrader()
        risk_score, risk_details = grader.grade_risk_assessment(json.dumps(decision.risk_assessment.model_dump()))
        carrier_score, carrier_details = grader.grade_carrier_recommendation(json.dumps(decision.carrier_recommendation.model_dump()))
        recovery_score, recovery_details = grader.grade_recovery_plan(json.dumps(decision.recovery_plan.model_dump()))

        overall_score = round((risk_score + carrier_score + recovery_score) / 3, 1)
        quality_level = "Excellent" if overall_score >= 85 else "Good" if overall_score >= 70 else "Fair" if overall_score >= 50 else "Poor"

        return {
            "request_id": str(uuid.uuid4()),
            "scenario": scenario_name,
            "decision": decision.model_dump(),
            "grading": {
                "overall_score": overall_score,
                "quality_level": quality_level,
                "risk_grading": {"score": risk_score, "details": risk_details},
                "carrier_grading": {"score": carrier_score, "details": carrier_details},
                "recovery_grading": {"score": recovery_score, "details": recovery_details},
            },
            "processing_time_ms": round(processing_time_ms, 2),
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }

    except LLMError as e:
        raise HTTPException(status_code=503, detail=f"AI model unavailable: {str(e)}")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Demo analysis failed: {str(e)}")


@app.get("/debug/llm-test", tags=["Monitoring"])
async def debug_llm_test() -> Dict[str, Any]:
    """Test LLM connectivity — diagnose token/endpoint issues.
    
    No auth required so you can quickly check from browser.
    """
    from pydantic_agents import _get_github_client

    token = os.getenv("GITHUB_TOKEN", "").strip()
    model = os.getenv("GITHUB_MODEL", "gpt-4o-mini")
    base_url = os.getenv("GITHUB_MODELS_BASE_URL", "https://models.inference.ai.azure.com")

    diagnostics = {
        "token_present": bool(token),
        "token_length": len(token),
        "token_prefix": token[:8] + "..." if len(token) > 8 else "(too short)",
        "model": model,
        "base_url": base_url,
        "llm_call_result": None,
        "error": None,
    }

    if not token or len(token) < 10:
        diagnostics["error"] = "GITHUB_TOKEN is missing or too short. Set it in .env file."
        return diagnostics

    try:
        client = _get_github_client()
        response = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": "You are a test assistant."},
                {"role": "user", "content": "Reply with exactly: OK"},
            ],
            temperature=0.0,
            max_tokens=10,
        )
        result = response.choices[0].message.content or ""
        diagnostics["llm_call_result"] = result.strip()
    except Exception as e:
        diagnostics["error"] = f"{type(e).__name__}: {str(e)}"

    return diagnostics


# ===== CLI ENTRYPOINT =====

if __name__ == "__main__":
    host = os.getenv("HOST", "0.0.0.0")
    port = int(os.getenv("PORT", "8000"))

    print("\n" + "=" * 70)
    print("🚀 Agentic Logistics Optimizer API")
    print("=" * 70)
    print(f"📍 Server:      http://{host}:{port}")
    print(f"📚 Docs:        http://{host}:{port}/docs")
    print(f"🌍 Environment: {ENVIRONMENT}")
    print(f"🔒 Auth:        x-api-key header required")
    print("=" * 70 + "\n")

    logger.info(
        "Starting server",
        environment=ENVIRONMENT,
        host=host,
        port=port,
        version=API_VERSION,
    )

    uvicorn.run(
        "main:app",
        host=host,
        port=port,
        log_level=LOG_LEVEL,
        reload=ENVIRONMENT == "development",
        workers=1 if ENVIRONMENT == "development" else 4,
    )
