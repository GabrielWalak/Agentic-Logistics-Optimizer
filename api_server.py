"""
PydanticAI FastAPI Server - Production-Grade
Enterprise-level REST API for Multi-Agent Logistics System
AWS Lambda + CloudWatch + DynamoDB Ready
"""

import os
import json
import uuid
import time
import logging
import asyncio
from datetime import datetime, timedelta, timezone
from typing import Optional, List, Dict, Any
from contextlib import asynccontextmanager
from functools import wraps

from fastapi import FastAPI, HTTPException, BackgroundTasks, Request, Header, Depends, Security
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.gzip import GZipMiddleware
from fastapi.responses import JSONResponse, StreamingResponse, HTMLResponse
from fastapi.exceptions import RequestValidationError
from fastapi.security import APIKeyHeader
from pydantic import BaseModel, Field, field_validator, ConfigDict
import uvicorn

# Import agent system
from pydantic_agents import (
    DeliveryScenario,
    IntegratedDecision,
    run_multi_agent_analysis_parallel,
    check_ollama_status,
)

# AWS integrations
try:
    import boto3
    AWS_AVAILABLE = True
except ImportError:
    AWS_AVAILABLE = False
    print("⚠ boto3 not installed - AWS integration disabled")

# ===== STRUCTURED LOGGING (CloudWatch Compatible) =====

class StructuredLogger:
    """JSON structured logger for CloudWatch Logs"""
    
    def __init__(self, name: str, level: int = logging.INFO):
        self.logger = logging.getLogger(name)
        self.logger.setLevel(level)
        
        if not self.logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter('%(message)s')
            handler.setFormatter(formatter)
            self.logger.addHandler(handler)
    
    def _log(self, level: str, message: str, **kwargs):
        """Log structured JSON to stdout (CloudWatch captures this)"""
        log_entry = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "level": level,
            "message": message,
            "service": "agentic-logistics-api",
            **kwargs
        }
        # Remove None values
        log_entry = {k: v for k, v in log_entry.items() if v is not None}
        self.logger.log(logging.INFO, json.dumps(log_entry))
    
    def info(self, message: str, **kwargs):
        self._log("INFO", message, **kwargs)
    
    def error(self, message: str, **kwargs):
        self._log("ERROR", message, **kwargs)
    
    def warning(self, message: str, **kwargs):
        self._log("WARNING", message, **kwargs)
    
    def debug(self, message: str, **kwargs):
        self._log("DEBUG", message, **kwargs)


logger = StructuredLogger(__name__)


# ===== API KEY SECURITY =====

API_KEY = os.getenv("API_KEY", "accenture2026")
api_key_header = APIKeyHeader(name="x-api-key", description="API Key for authentication")

async def verify_api_key(api_key: str = Security(api_key_header)) -> str:
    """Verify API key from request header"""
    if api_key != API_KEY:
        logger.warning("Invalid API key attempt", provided_key_prefix=api_key[:4] if api_key else "none")
        raise HTTPException(status_code=403, detail="Invalid API key")
    return api_key


# ===== AWS CLIENTS =====

class AWSManager:
    """Manage AWS services (DynamoDB, CloudWatch, S3)"""
    
    _instance = None
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._initialized = False
        return cls._instance
    
    def __init__(self):
        if self._initialized:
            return
        
        self.available = AWS_AVAILABLE
        self.dynamodb = None
        self.cloudwatch = None
        self.s3 = None
        
        if AWS_AVAILABLE:
            try:
                region = os.getenv("AWS_REGION", "eu-central-1")
                self.dynamodb = boto3.resource("dynamodb", region_name=region)
                self.cloudwatch = boto3.client("cloudwatch", region_name=region)
                self.s3 = boto3.client("s3", region_name=region)
                logger.info("AWS services initialized", region=region)
            except Exception as e:
                logger.error("AWS initialization failed", error=str(e))
                self.available = False
        
        self._initialized = True
    
    def put_audit_log(self, request_id: str, data: Dict[str, Any]):
        """Store decision in DynamoDB"""
        if not self.available or not self.dynamodb:
            return
        
        try:
            table_name = os.getenv("DYNAMODB_TABLE", "agentic-logistics-decisions")
            table = self.dynamodb.Table(table_name)
            
            item = {
                "request_id": request_id,
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "ttl": int((datetime.now(timezone.utc) + timedelta(days=90)).timestamp()),
                "data": json.dumps(data),
                "created_at": datetime.now(timezone.utc).isoformat(),
            }
            
            table.put_item(Item=item)
            logger.info("Audit log stored", request_id=request_id, table=table_name)
        except Exception as e:
            logger.error("DynamoDB write failed", request_id=request_id, error=str(e))
    
    def publish_metric(self, metric_name: str, value: float, unit: str = "None"):
        """Publish metric to CloudWatch"""
        if not self.available or not self.cloudwatch:
            return
        
        try:
            self.cloudwatch.put_metric_data(
                Namespace="AgenticLogistics",
                MetricData=[
                    {
                        "MetricName": metric_name,
                        "Value": value,
                        "Unit": unit,
                        "Timestamp": datetime.now(timezone.utc),
                    }
                ],
            )
        except Exception as e:
            logger.error("CloudWatch metric failed", metric=metric_name, error=str(e))


aws_manager = AWSManager()


# ===== REQUEST/RESPONSE MODELS =====

class AnalysisRequest(BaseModel):
    """HTTP request for delivery analysis"""
    predicted_days: float = Field(..., gt=0, description="Predicted delivery days")
    promised_days: float = Field(default=7.0, gt=0, description="Promised delivery days")
    distance_km: float = Field(..., gt=0, description="Distance in km")
    weight_g: float = Field(..., gt=0, description="Package weight in grams")
    payment_lag_days: int = Field(default=0, ge=0, description="Days between order and shipment")
    is_weekend_order: int = Field(default=0, ge=0, le=1, description="1 if weekend, 0 otherwise")
    freight_value: float = Field(..., ge=0, description="Freight cost in BRL")
    rag_context: str = Field(
        default="Standard carrier rules apply",
        max_length=2000,
        description="Knowledge base context"
    )
    
    @field_validator("rag_context")
    @classmethod
    def validate_context(cls, v):
        """Ensure context is not empty"""
        if not v or not v.strip():
            return "Standard carrier rules apply"
        return v.strip()


class ErrorResponse(BaseModel):
    """Standardized error response"""
    status: str = "error"
    request_id: str
    error: str
    error_code: str
    timestamp: str
    details: Optional[Dict[str, Any]] = None


class AnalysisResponse(BaseModel):
    """HTTP response from analysis"""
    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "status": "success",
                "request_id": "abc12345",
                "timestamp": "2026-04-29T10:30:00.000Z",
                "processing_time_ms": 8234.5,
                "decision": {
                    "risk_assessment": {"risk_level": "HIGH", "risk_score": 72.5},
                    "carrier_recommendation": {"recommended_carrier": "Premium Express"},
                    "recovery_plan": {"voucher_code": "DELAY25"},
                    "confidence_score": 85.0,
                }
            }
        }
    )
    
    status: str = Field(default="success", description="success | error")
    request_id: str = Field(description="Unique request ID for tracing")
    timestamp: str = Field(description="ISO 8601 timestamp")
    processing_time_ms: float = Field(description="Time in milliseconds")
    
    # Decision data (only if success)
    decision: Optional[Dict[str, Any]] = None
    
    # Error info (only if error)
    error: Optional[str] = None
    error_code: Optional[str] = None


class HealthResponse(BaseModel):
    """Health check response"""
    status: str
    timestamp: str
    version: str
    service_name: str
    llm_ready: bool
    aws_available: bool
    cache_enabled: bool
    environment: str


class MetricsResponse(BaseModel):
    """Metrics snapshot"""
    total_requests: int
    successful: int
    failed: int
    average_processing_time_ms: float
    timestamp: str


# ===== GLOBAL STATE =====

class AppState:
    """Application state management"""
    
    def __init__(self):
        self.start_time = datetime.now(timezone.utc)
        self.request_count = 0
        self.success_count = 0
        self.error_count = 0
        self.total_processing_time = 0.0
        self.lock = asyncio.Lock()
    
    async def increment_request(self, processing_time_ms: float, success: bool):
        """Update request statistics"""
        async with self.lock:
            self.request_count += 1
            self.total_processing_time += processing_time_ms
            if success:
                self.success_count += 1
            else:
                self.error_count += 1


app_state = AppState()


DEMO_ANALYSIS_PAYLOAD: Dict[str, Any] = {}


def build_home_page(base_url: str, demo: Dict[str, Any], api_key: str = API_KEY) -> str:
    """Render the console-style home page with a precomputed demo analysis."""
    risk = demo.get("risk_assessment", {})
    carrier = demo.get("carrier_recommendation", {})
    recovery = demo.get("recovery_plan", {})

    factors = risk.get("primary_risk_factors") or []
    factors_preview = ", ".join(factors[:2]) if factors else "No risk factors returned"
    executive_summary = demo.get("executive_summary", "No executive summary available")

    return f"""
    <!DOCTYPE html>
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
            .warn {{ color: #f48771; }}
            .security {{ background: #3d2d2d; border-left: 4px solid #f48771; padding: 10px; margin: 10px 0; }}
            a {{ color: #569cd6; text-decoration: none; }}
            a:hover {{ text-decoration: underline; }}
        </style>
    </head>
    <body>
        <div class="container">
            <h1>🚀 Agentic Logistics Optimizer API</h1>
            <p>Enterprise-grade multi-agent system for logistics decision-making</p>
            <p><strong>Version:</strong> 1.0.0 | <strong>Status:</strong> Running ✓</p>

            <div class="panel">
                <h2>▶ Console-style Demo Analysis</h2>
                <div><span class="label">Risk level:</span> <span class="value">{risk.get('risk_level', 'N/A')}</span></div>
                <div><span class="label">Risk score:</span> <span class="value">{risk.get('risk_score', 'N/A')}/100</span></div>
                <div><span class="label">Carrier:</span> <span class="value">{carrier.get('recommended_carrier', 'N/A')}</span></div>
                <div><span class="label">Upgrade needed:</span> <span class="value">{'Yes' if carrier.get('should_upgrade') else 'No'}</span></div>
                <div><span class="label">Voucher:</span> <span class="value">{recovery.get('voucher_code') or 'None'}</span></div>
                <div><span class="label">Confidence:</span> <span class="value">{demo.get('confidence_score', 'N/A')}/100</span></div>
                <div style="margin-top: 10px;"><span class="label">Factors:</span> <span class="value">{factors_preview}</span></div>
                <div style="margin-top: 10px;"><span class="label">Summary:</span> <span class="value">{executive_summary}</span></div>
            </div>

            <div class="security">
                <h3 style="color: #f48771; margin-top: 0;">🔒 Authentication Required</h3>
                <p>All API endpoints require authentication via <strong>x-api-key</strong> header.</p>
                <p><strong>Default key:</strong> <code>{api_key}</code></p>
                <p>Set <strong>API_KEY</strong> environment variable to customize.</p>
            </div>

            <h2>📚 Documentation</h2>
            <ul>
                <li><a href="{base_url}/docs">Interactive API Docs (Swagger) - with Authorize button</a></li>
                <li><a href="{base_url}/redoc">ReDoc Documentation</a></li>
                <li><a href="{base_url}/health">Health Check</a></li>
                <li><a href="{base_url}/status">Service Status</a></li>
            </ul>

            <h2>🛠️ Quick Commands (with API Key)</h2>
            <div class="endpoint">
                <div><span class="method">GET</span> <span class="url">/health</span></div>
                <div class="example">curl {base_url}/health</div>
            </div>
            <div class="endpoint">
                <div><span class="method">POST</span> <span class="url">/analyze</span> (requires auth)</div>
                <div class="example">curl -X POST {base_url}/analyze \\
  -H "Content-Type: application/json" \\
  -H "x-api-key: {api_key}" \\
  -d '{{"predicted_days": 8.5, "promised_days": 7, "distance_km": 450, "weight_g": 1200, "freight_value": 45.0}}'</div>
            </div>
            <div class="endpoint">
                <div><span class="method">POST</span> <span class="url">/batch-analyze</span> (requires auth)</div>
                <div class="example">curl -X POST {base_url}/batch-analyze \\
  -H "Content-Type: application/json" \\
  -H "x-api-key: {api_key}" \\
  -d '[{{"predicted_days": 8.5, "promised_days": 7, "distance_km": 450, "weight_g": 1200, "freight_value": 45.0}}]'</div>
            </div>

            <h2>📊 Last Demo Output</h2>
            <div class="panel">
                <div><span class="label">Estimated delivery:</span> <span class="value">{demo.get('estimated_delivery_time', 'N/A')} days</span></div>
                <div><span class="label">Processing mode:</span> <span class="value">cached demo</span></div>

            </div>

            <h2>✨ Features</h2>
            <ul>
                <li>✓ Multi-agent AI (Risk, Carrier, Recovery, Orchestrator)</li>
                <li>✓ 50+ concurrent requests support</li>
                <li>✓ Redis caching</li>
                <li>✓ AWS Lambda ready</li>
                <li>✓ Real-time analysis endpoint</li>
            </ul>
        </div>
    </body>
    </html>
    """


# ===== STARTUP/SHUTDOWN =====

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifecycle manager"""
    # Startup
    logger.info("API Server starting up", version="1.0.0")
    
    # Check LLM availability
    is_llm_ready = check_ollama_status()
    if not is_llm_ready:
        logger.warning("LLM not available at startup - will retry on requests")
    
    app.llm_ready = is_llm_ready
    app_state.start_time = datetime.now(timezone.utc)
    
    global DEMO_ANALYSIS_PAYLOAD
    DEMO_ANALYSIS_PAYLOAD = {
        "risk_assessment": {
            "risk_level": "MODERATE",
            "risk_score": 65.0,
            "primary_risk_factors": [
                "Predicted Delivery vs. Promised Window",
                "Distance",
                "Weight",
                "Payment Lag",
            ],
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
        "executive_summary": (
            "Example analysis: the delivery is slightly late, so the system recommends "
            "a regional carrier upgrade and a DELAY25 voucher to reduce risk and protect retention."
        ),
        "estimated_delivery_time": 7.0,
        "confidence_score": 78.0,
    }
    logger.info(
        "Cached demo analysis ready",
        confidence_score=DEMO_ANALYSIS_PAYLOAD.get("confidence_score"),
        demo_mode="static",
    )
    
    yield
    
    # Shutdown
    logger.info(
        "API Server shutting down",
        total_requests=app_state.request_count,
        uptime_seconds=(datetime.now(timezone.utc) - app_state.start_time).total_seconds(),
    )


# ===== FASTAPI APP SETUP =====

from fastapi.openapi.utils import get_openapi

app = FastAPI(
    title="Agentic Logistics Optimizer API",
    description="Enterprise-grade multi-agent system for logistics decision-making",
    version="1.0.0",
    lifespan=lifespan,
)

# Custom OpenAPI schema with security schemes
def custom_openapi():
    if app.openapi_schema:
        return app.openapi_schema
    
    openapi_schema = get_openapi(
        title="Agentic Logistics Optimizer API",
        version="1.0.0",
        description="Enterprise-grade multi-agent system for logistics decision-making",
        routes=app.routes,
    )
    
    openapi_schema["components"]["securitySchemes"] = {
        "APIKeyHeader": {
            "type": "apiKey",
            "in": "header",
            "name": "x-api-key",
            "description": "API Key for authentication"
        }
    }
    
    app.openapi_schema = openapi_schema
    return app.openapi_schema

app.openapi = custom_openapi

# CORS - Adjust origins for production
origins = os.getenv("CORS_ORIGINS", "*").split(",")
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Compression
app.add_middleware(GZipMiddleware, minimum_size=1000)


# ===== CUSTOM EXCEPTION HANDLERS =====

@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request: Request, exc: RequestValidationError):
    """Handle Pydantic validation errors"""
    request_id = request.state.request_id
    
    errors = []
    for error in exc.errors():
        errors.append({
            "field": ".".join(str(x) for x in error["loc"][1:]),
            "type": error["type"],
            "message": error["msg"],
        })
    
    logger.error(
        "Validation error",
        request_id=request_id,
        errors=errors,
    )
    
    return JSONResponse(
        status_code=422,
        content=ErrorResponse(
            request_id=request_id,
            error="Request validation failed",
            error_code="VALIDATION_ERROR",
            timestamp=datetime.now(timezone.utc).isoformat(),
            details={"validation_errors": errors},
        ).dict(),
    )


@app.exception_handler(Exception)
async def general_exception_handler(request: Request, exc: Exception):
    """Handle uncaught exceptions"""
    request_id = getattr(request.state, "request_id", "unknown")
    
    logger.error(
        "Unhandled exception",
        request_id=request_id,
        error_type=type(exc).__name__,
        error=str(exc),
    )
    
    return JSONResponse(
        status_code=500,
        content=ErrorResponse(
            request_id=request_id,
            error="Internal server error",
            error_code="INTERNAL_ERROR",
            timestamp=datetime.now(timezone.utc).isoformat(),
        ).dict(),
    )


# ===== MIDDLEWARE =====

@app.middleware("http")
async def add_request_id_middleware(request: Request, call_next):
    """Add request ID to all requests"""
    request_id = str(uuid.uuid4())[:8]
    request.state.request_id = request_id
    
    start_time = time.perf_counter()
    response = await call_next(request)
    process_time = (time.perf_counter() - start_time) * 1000  # ms
    
    response.headers["X-Request-ID"] = request_id
    response.headers["X-Process-Time"] = str(process_time)
    
    logger.info(
        "HTTP request",
        request_id=request_id,
        method=request.method,
        path=request.url.path,
        status_code=response.status_code,
        process_time_ms=round(process_time, 2),
    )
    
    return response


# ===== DEPENDENCY INJECTIONS =====

async def get_request_id(request: Request) -> str:
    """Extract request ID from context"""
    return request.state.request_id


async def rate_limit_check(
    request: Request,
    x_api_key: Optional[str] = Header(None),
) -> bool:
    """Simple rate limiting (can use Redis for distributed)"""
    # Placeholder - extend with actual rate limiting
    return True


# ===== HEALTH & MONITORING ENDPOINTS =====

@app.get("/", tags=["Info"])
async def root(request: Request):
    """Welcome endpoint with cached demo analysis"""
    try:
        host = request.url.hostname or "localhost"
        port = request.url.port or 8000
        base_url = f"http://{host}:{port}"
        html_content = build_home_page(base_url, DEMO_ANALYSIS_PAYLOAD)
        logger.info("Root endpoint accessed", host=host, port=port, content_length=len(html_content))
        return HTMLResponse(content=html_content)
    except Exception as e:
        logger.error("Root endpoint error", error=str(e), error_type=type(e).__name__)
        import traceback
        print(f"ERROR in root(): {traceback.format_exc()}")
        raise HTTPException(status_code=500, detail=f"Error rendering home page: {str(e)}")
    
    

@app.get("/health", response_model=HealthResponse, tags=["Monitoring"])
async def health_check(request_id: str = Depends(get_request_id)):
    """Health check endpoint"""
    
    # Check LLM
    llm_ready = app.llm_ready if hasattr(app, "llm_ready") else check_ollama_status()
    
    response = HealthResponse(
        status="healthy",
        timestamp=datetime.now(timezone.utc).isoformat(),
        version="1.0.0",
        service_name="agentic-logistics-api",
        llm_ready=llm_ready,
        aws_available=aws_manager.available,
        cache_enabled=True,  # Set based on your cache
        environment=os.getenv("ENVIRONMENT", "development"),
    )
    
    logger.info("Health check", request_id=request_id, status=response.status)
    return response


@app.get("/status", tags=["Monitoring"])
async def service_status(request_id: str = Depends(get_request_id)):
    """Detailed service status"""
    from pydantic_agents import rag_cache, LANGSMITH_AVAILABLE
    
    uptime = datetime.now(timezone.utc) - app_state.start_time
    avg_time = (
        app_state.total_processing_time / app_state.request_count
        if app_state.request_count > 0
        else 0
    )
    
    return {
        "status": "operational",
        "uptime_seconds": uptime.total_seconds(),
        "request_count": app_state.request_count,
        "success_count": app_state.success_count,
        "error_count": app_state.error_count,
        "average_processing_time_ms": round(avg_time, 2),
        "model": os.getenv("GITHUB_MODEL", "gpt-4o-mini"),
        "cache_enabled": rag_cache.enabled,
        "langsmith_enabled": LANGSMITH_AVAILABLE,
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }


@app.get("/metrics", response_model=MetricsResponse, tags=["Monitoring"])
async def get_metrics(request_id: str = Depends(get_request_id)):
    """Get service metrics"""
    avg_time = (
        app_state.total_processing_time / app_state.request_count
        if app_state.request_count > 0
        else 0
    )
    
    return MetricsResponse(
        total_requests=app_state.request_count,
        successful=app_state.success_count,
        failed=app_state.error_count,
        average_processing_time_ms=round(avg_time, 2),
        timestamp=datetime.now(timezone.utc).isoformat(),
    )


# ===== MAIN ANALYSIS ENDPOINT =====

@app.post(
    "/analyze",
    response_model=AnalysisResponse,
    tags=["Analysis"],
    summary="Analyze delivery scenario",
    responses={
        200: {"description": "Successful analysis"},
        422: {"description": "Validation error"},
        503: {"description": "Service unavailable"},
    },
)
async def analyze_delivery(
    request_body: AnalysisRequest,
    background_tasks: BackgroundTasks,
    request: Request,
    request_id: str = Depends(get_request_id),
    _: bool = Depends(rate_limit_check),
    __: str = Depends(verify_api_key),
) -> AnalysisResponse:
    """
    Analyze delivery scenario and get integrated multi-agent decision
    
    Returns:
    - Risk assessment (risk level, score, mitigation priority)
    - Carrier recommendation (upgrade decision, cost impact, ROI)
    - Customer recovery plan (voucher, discount, timing)
    - Executive summary with confidence score
    
    Example response:
    ```json
    {
        "status": "success",
        "request_id": "a1b2c3d4",
        "decision": {
            "risk_assessment": {
                "risk_level": "HIGH",
                "risk_score": 72.5
            }
        }
    )
    ```
    """
    
    start_time = time.perf_counter()
    
    try:
        logger.info(
            "Analysis request received",
            request_id=request_id,
            predicted_days=request_body.predicted_days,
            distance_km=request_body.distance_km,
        )
        
        # Build delivery scenario
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
        
        # Run multi-agent analysis
        decision: IntegratedDecision = run_multi_agent_analysis_parallel(scenario)
        
        processing_time_ms = (time.perf_counter() - start_time) * 1000
        
        # Prepare response
        response_data = {
            "status": "success",
            "request_id": request_id,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "processing_time_ms": round(processing_time_ms, 2),
            "decision": decision.dict(),
        }
        
        # Background tasks
        background_tasks.add_task(
            _audit_decision,
            request_id=request_id,
            request_data=request_body.dict(),
            decision_data=decision.dict(),
            processing_time_ms=processing_time_ms,
        )
        
        # Update metrics
        await app_state.increment_request(processing_time_ms, success=True)
        
        # Publish metrics
        aws_manager.publish_metric("AnalysisSuccess", 1, "Count")
        aws_manager.publish_metric("ProcessingTime", processing_time_ms, "Milliseconds")
        
        logger.info(
            "Analysis completed successfully",
            request_id=request_id,
            processing_time_ms=round(processing_time_ms, 2),
            confidence_score=decision.confidence_score,
        )
        
        return AnalysisResponse(**response_data)
    
    except ValueError as e:
        processing_time_ms = (time.perf_counter() - start_time) * 1000
        await app_state.increment_request(processing_time_ms, success=False)
        aws_manager.publish_metric("AnalysisValidationError", 1, "Count")
        
        logger.error(
            "Validation error in analysis",
            request_id=request_id,
            error=str(e),
        )
        
        raise HTTPException(status_code=422, detail=str(e))
    
    except Exception as e:
        processing_time_ms = (time.perf_counter() - start_time) * 1000
        await app_state.increment_request(processing_time_ms, success=False)
        aws_manager.publish_metric("AnalysisError", 1, "Count")
        
        logger.error(
            "Unexpected error in analysis",
            request_id=request_id,
            error_type=type(e).__name__,
            error=str(e),
        )
        
        raise HTTPException(status_code=500, detail="Analysis failed")


# ===== BATCH PROCESSING ENDPOINT =====

@app.post("/batch-analyze", tags=["Analysis"])
async def batch_analyze(
    requests: List[AnalysisRequest],
    background_tasks: BackgroundTasks,
    request_id: str = Depends(get_request_id),
    _: bool = Depends(rate_limit_check),
    __: str = Depends(verify_api_key),
):
    """
    Analyze multiple delivery scenarios in parallel
    
    Useful for dashboards, reports, and bulk processing
    """
    
    if not requests:
        raise HTTPException(status_code=400, detail="Empty request list")
    
    if len(requests) > 100:
        raise HTTPException(
            status_code=400,
            detail="Maximum 100 scenarios per batch"
        )
    
    start_time = time.perf_counter()
    results = []
    
    logger.info(
        "Batch analysis started",
        request_id=request_id,
        scenario_count=len(requests),
    )
    
    try:
        # Process in parallel with semaphore to limit concurrency
        semaphore = asyncio.Semaphore(5)  # Max 5 concurrent
        
        async def process_one(idx: int, req: AnalysisRequest):
            async with semaphore:
                try:
                    scenario = DeliveryScenario(**req.dict())
                    decision = run_multi_agent_analysis_parallel(scenario)
                    return {
                        "index": idx,
                        "status": "success",
                        "decision": decision.dict(),
                    }
                except Exception as e:
                    logger.error(
                        "Batch item failed",
                        request_id=request_id,
                        index=idx,
                        error=str(e),
                    )
                    return {
                        "index": idx,
                        "status": "error",
                        "error": str(e),
                    }
        
        # Create tasks
        tasks = [process_one(idx, req) for idx, req in enumerate(requests)]
        results = await asyncio.gather(*tasks)
        
        processing_time_ms = (time.perf_counter() - start_time) * 1000
        
        # Log results
        successful = sum(1 for r in results if r["status"] == "success")
        
        logger.info(
            "Batch analysis completed",
            request_id=request_id,
            total=len(requests),
            successful=successful,
            failed=len(requests) - successful,
            processing_time_ms=round(processing_time_ms, 2),
        )
        
        await app_state.increment_request(processing_time_ms, success=True)
        
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
        
        logger.error(
            "Batch analysis failed",
            request_id=request_id,
            error=str(e),
        )
        
        raise HTTPException(status_code=500, detail="Batch processing failed")


# ===== BACKGROUND TASKS =====

def _audit_decision(
    request_id: str,
    request_data: Dict[str, Any],
    decision_data: Dict[str, Any],
    processing_time_ms: float,
):
    """Store decision in audit trail (DynamoDB)"""
    try:
        audit_entry = {
            "request_id": request_id,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "request": request_data,
            "decision": decision_data,
            "processing_time_ms": processing_time_ms,
        }
        
        # Store in DynamoDB
        aws_manager.put_audit_log(request_id, audit_entry)
        
        logger.info("Audit log stored", request_id=request_id)
    
    except Exception as e:
        logger.error("Audit storage failed", request_id=request_id, error=str(e))


# ===== AWS LAMBDA ADAPTER =====

async def lambda_handler(event, context):
    """AWS Lambda entry point using Mangum"""
    from mangum import Mangum
    
    handler = Mangum(app, lifespan="auto")
    return await handler(event, context)


# ===== CLI ENTRY POINT =====

if __name__ == "__main__":
    environment = os.getenv("ENVIRONMENT", "development")
    host = os.getenv("HOST", "0.0.0.0")
    port = int(os.getenv("PORT", "8000"))
    
    logger.info(
        "Starting Agentic Logistics API",
        environment=environment,
        host=host,
        port=port,
    )
    
    print("\n" + "=" * 60)
    print("🚀 Agentic Logistics Optimizer API")
    print("=" * 60)
    print(f"📍 Server: http://{host}:{port}")
    print(f"📚 Docs:   http://{host}:{port}/docs")
    print(f"🔄 ReDoc:  http://{host}:{port}/redoc")
    print(f"🌍 Environment: {environment}")
    print("=" * 60 + "\n")
    
    uvicorn.run(
        app,
        host=host,
        port=port,
        log_level=os.getenv("LOG_LEVEL", "info").lower(),
        reload=environment == "development",
        workers=1 if environment == "development" else 4,
    )
