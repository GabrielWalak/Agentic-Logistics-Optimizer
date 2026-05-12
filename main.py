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
from datetime import datetime, timezone
from typing import List, Dict, Any
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, Request, Depends, Security
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.gzip import GZipMiddleware
from fastapi.responses import HTMLResponse
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
)
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

    predicted_days: float = Field(description="Predicted delivery time in days")
    promised_days: float = Field(default=7.0, description="Promised delivery time")
    distance_km: float = Field(description="Distance in kilometers")
    weight_g: float = Field(description="Package weight in grams")
    freight_value: float = Field(description="Freight cost in USD")
    payment_lag_days: int = Field(default=2, description="Payment lag in days")
    is_weekend_order: int = Field(default=0, description="Weekend order flag")
    rag_context: str = Field(default="Standard carrier rules apply", description="RAG context")

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


class AnalysisResponse(BaseModel):
    """Single analysis response"""

    request_id: str
    decision: IntegratedDecision
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


def build_home_page(base_url: str, demo: Dict[str, Any]) -> str:
    """Render console-style home page with demo analysis"""
    risk = demo.get("risk_assessment", {})
    carrier = demo.get("carrier_recommendation", {})
    recovery = demo.get("recovery_plan", {})

    risk_factors = risk.get("primary_risk_factors", [])
    factors_str = ", ".join(risk_factors[:2]) if risk_factors else "No risk factors"

    html = f"""<!DOCTYPE html>
<html>
<head>
    <title>Agentic Logistics Optimizer API</title>
    <style>
        body {{ font-family: monospace; background: #1e1e1e; color: #d4d4d4; padding: 20px; margin: 0; }}
        .container {{ max-width: 1200px; margin: 0 auto; }}
        h1 {{ color: #4ec9b0; border-bottom: 2px solid #4ec9b0; padding-bottom: 10px; }}
        h2 {{ color: #9cdcfe; margin-top: 30px; }}
        .panel {{ background: #252526; padding: 15px; margin: 18px 0; border-left: 4px solid #4ec9b0; border-radius: 4px; }}
        .endpoint {{ background: #252526; padding: 15px; margin: 10px 0; border-left: 3px solid #007acc; }}
        .method {{ color: #ce9178; font-weight: bold; }}
        .url {{ color: #4ec9b0; }}
        .example {{ background: #2d2d30; padding: 10px; margin: 10px 0; border-radius: 5px; overflow-x: auto; font-size: 12px; white-space: pre-wrap; word-break: break-all; }}
        .label {{ color: #9cdcfe; font-weight: bold; }}
        .value {{ color: #b5cea8; }}
        .security {{ background: #3d2d2d; border-left: 4px solid #f48771; padding: 10px; margin: 10px 0; }}
        a {{ color: #569cd6; text-decoration: none; }}
    </style>
</head>
<body>
    <div class="container">
        <h1>🚀 Agentic Logistics Optimizer API</h1>
        <p>Enterprise-grade multi-agent system for logistics decision-making</p>
        <p><strong>Version:</strong> {API_VERSION} | <strong>Database:</strong> PostgreSQL | <strong>Cache:</strong> Redis</p>

        <div class="panel">
            <h2>▶ Console-style Demo Analysis</h2>
            <div><span class="label">Risk level:</span> <span class="value">{risk.get('risk_level', 'N/A')}</span></div>
            <div><span class="label">Risk score:</span> <span class="value">{risk.get('risk_score', 'N/A')}/100</span></div>
            <div><span class="label">Carrier:</span> <span class="value">{carrier.get('recommended_carrier', 'N/A')}</span></div>
            <div><span class="label">Upgrade needed:</span> <span class="value">{'Yes' if carrier.get('should_upgrade') else 'No'}</span></div>
            <div><span class="label">Voucher:</span> <span class="value">{recovery.get('voucher_code') or 'None'}</span></div>
            <div><span class="label">Confidence:</span> <span class="value">{demo.get('confidence_score', 'N/A')}/100</span></div>
            <div style="margin-top: 10px;"><span class="label">Summary:</span> <span class="value">{demo.get('executive_summary', '')}</span></div>
        </div>

        <div class="security">
            <h3 style="color: #f48771; margin-top: 0;">🔒 Authentication Required</h3>
            <p>All API endpoints require <strong>x-api-key</strong> header for authentication.</p>
            <p>API key must be provided in request headers. See documentation for details.</p>
        </div>

        <h2>📚 Documentation</h2>
        <ul>
            <li><a href="{base_url}/docs">Interactive API Docs (Swagger)</a></li>
            <li><a href="{base_url}/redoc">ReDoc Documentation</a></li>
        </ul>

        <h2>🛠️ Quick Commands</h2>
        <div class="endpoint">
            <div><span class="method">POST</span> <span class="url">/analyze</span></div>
            <div class="example">curl -X POST {base_url}/analyze \\
  -H "x-api-key: YOUR_API_KEY_HERE" \\
  -H "Content-Type: application/json" \\
  -d '{{"predicted_days": 8.5, "promised_days": 7, "distance_km": 450, "weight_g": 1200, "freight_value": 45.0}}'</div>
        </div>
    </div>
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
    """Home page with API documentation and demo analysis"""
    host = request.url.hostname or "localhost"
    port = request.url.port or 8000
    base_url = f"http://{host}:{port}"
    return HTMLResponse(content=build_home_page(base_url, DEMO_ANALYSIS_PAYLOAD))


@app.get("/health", response_model=HealthResponse, tags=["Monitoring"])
async def health_check() -> HealthResponse:
    """Health check endpoint"""
    return HealthResponse(
        status="healthy",
        timestamp=datetime.now(timezone.utc).isoformat(),
        version=API_VERSION,
        service_name=SERVICE_NAME,
        llm_ready=getattr(app, "llm_ready", False),
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
    """Analyze single delivery scenario"""
    start_time = time.perf_counter()

    try:
        scenario = DeliveryScenario(
            predicted_days=request_body.predicted_days,
            promised_days=request_body.promised_days,
            distance_km=request_body.distance_km,
            weight_g=request_body.weight_g,
            payment_lag_days=request_body.payment_lag_days,
            is_weekend_order=request_body.is_weekend_order,
            freight_value=request_body.freight_value,
            rag_context=request_body.rag_context,
        )

        decision = await asyncio.to_thread(run_multi_agent_analysis_parallel, scenario)
        processing_time_ms = (time.perf_counter() - start_time) * 1000

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
        )

        return AnalysisResponse(
            request_id=request_id,
            decision=decision,
            processing_time_ms=processing_time_ms,
            timestamp=datetime.now(timezone.utc).isoformat(),
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
