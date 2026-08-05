"""FastAPI entry point for the multi-agent logistics portfolio project."""

import os
import json
import uuid
import time
import logging
import asyncio
import base64
from datetime import datetime, timezone
from typing import List, Dict, Any, Optional
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, Request, Depends, Security
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.gzip import GZipMiddleware
from fastapi.responses import HTMLResponse, Response
from fastapi.security import APIKeyHeader
from fastapi.openapi.utils import get_openapi
from pydantic import BaseModel, Field, ConfigDict
from sqlalchemy.ext.asyncio import AsyncSession
import uvicorn

from pydantic_agents import (
    DeliveryScenario,
    IntegratedDecision,
    run_multi_agent_analysis_parallel,
    LLMError,
    rag_cache,
    get_llm_config,
    _get_llm_client,
    build_deterministic_fallback_decision,
)
from prompt_engineering import ResponseGrader
from ml_predictor import predict_delivery_days, get_model_info
from models import AuditLog
from database import check_database_health, get_session, init_db


# ===== CONFIGURATION =====

API_KEY = os.getenv("API_KEY", "change-me-in-production-use-env-variable")
ENVIRONMENT = os.getenv("ENVIRONMENT", "development")
CORS_ORIGINS = os.getenv("CORS_ORIGINS", "*").split(",")
LOG_LEVEL = os.getenv("LOG_LEVEL", "info").lower()
SERVICE_NAME = "agentic-logistics-api"
API_VERSION = "1.0.0"
DEFAULT_RAG_CONTEXT = "Standard carrier rules apply"
ANALYSIS_TIMEOUT_SECONDS = float(os.getenv("ANALYSIS_TIMEOUT_SECONDS", "75"))
HEALTH_CHECK_TIMEOUT_SECONDS = float(os.getenv("HEALTH_CHECK_TIMEOUT_SECONDS", "2"))


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
    rag_context: str = Field(default=DEFAULT_RAG_CONTEXT, description="RAG context (auto-retrieved if not provided)")

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
    database_ready: bool
    ml_model_ready: bool
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


def _prepare_analysis_scenario(request_body: AnalysisRequest) -> DeliveryScenario:
    """Resolve optional ML and RAG inputs into one validated scenario.

    This synchronous preparation is shared by single and batch endpoints and is
    executed in a worker thread by callers. Keeping it in one place prevents
    the two API paths from silently applying different business rules.
    """

    predicted_days = request_body.predicted_days
    if predicted_days is None or predicted_days <= 0:
        predicted_days = predict_delivery_days(
            distance_km=request_body.distance_km,
            weight_g=request_body.weight_g,
            freight_value=request_body.freight_value,
            payment_lag_days=request_body.payment_lag_days,
            is_weekend_order=request_body.is_weekend_order,
        )
    if predicted_days is None:
        predicted_days = 7.0

    rag_context = request_body.rag_context
    if rag_context == DEFAULT_RAG_CONTEXT:
        rag_context = _get_rag_context(
            distance_km=request_body.distance_km,
            weight_g=request_body.weight_g,
            payment_lag_days=request_body.payment_lag_days,
            is_weekend_order=request_body.is_weekend_order,
            predicted_days=predicted_days,
            promised_days=request_body.promised_days,
        )

    return DeliveryScenario(
        predicted_days=predicted_days,
        promised_days=request_body.promised_days,
        distance_km=request_body.distance_km,
        weight_g=request_body.weight_g,
        payment_lag_days=request_body.payment_lag_days,
        is_weekend_order=request_body.is_weekend_order,
        freight_value=request_body.freight_value,
        rag_context=rag_context,
    )


async def _run_analysis_with_timeout(
    scenario: DeliveryScenario,
) -> IntegratedDecision:
    """Run the blocking agent workflow without blocking the FastAPI event loop."""

    return await asyncio.wait_for(
        asyncio.to_thread(run_multi_agent_analysis_parallel, scenario),
        timeout=ANALYSIS_TIMEOUT_SECONDS,
    )


def _grade_decision(decision: IntegratedDecision) -> GradingResult:
    """Apply the deterministic quality rubric to normalized agent outputs."""

    grader = ResponseGrader()
    risk_score, risk_details = grader.grade_risk_assessment(
        json.dumps(decision.risk_assessment.model_dump())
    )
    carrier_score, carrier_details = grader.grade_carrier_recommendation(
        json.dumps(decision.carrier_recommendation.model_dump())
    )
    recovery_score, recovery_details = grader.grade_recovery_plan(
        json.dumps(decision.recovery_plan.model_dump())
    )

    overall_score = round((risk_score + carrier_score + recovery_score) / 3, 1)
    if overall_score >= 85:
        quality_level = "Excellent"
    elif overall_score >= 70:
        quality_level = "Good"
    elif overall_score >= 50:
        quality_level = "Fair"
    else:
        quality_level = "Poor"

    return GradingResult(
        overall_score=overall_score,
        quality_level=quality_level,
        risk_grading={"score": risk_score, "details": risk_details},
        carrier_grading={"score": carrier_score, "details": carrier_details},
        recovery_grading={"score": recovery_score, "details": recovery_details},
    )


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


def require_auth_response() -> Response:
    """Return 401 with WWW-Authenticate header to trigger browser login popup."""
    return Response(
        content="Authentication required. Use any username with the portfolio password.",
        status_code=401,
        headers={"WWW-Authenticate": 'Basic realm="Portfolio Access"'},
        media_type="text/plain",
    )


from templates import build_home_page


# ===== APPLICATION INSTANCE & STATE =====

app_state = AppState()


@asynccontextmanager
async def lifespan(application: FastAPI):
    """Application lifecycle management"""
    logger.info("Starting API server", version=API_VERSION, environment=ENVIRONMENT)

    # Initialize database
    try:
        await init_db()
        logger.info("Database initialized successfully")
        application.state.db_ready = True
    except Exception as e:
        logger.warning("Database initialization failed (running in demo mode)", error=str(e))
        application.state.db_ready = False

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
    description="Portfolio multi-agent workflow for logistics decision support",
    version=API_VERSION,
    lifespan=lifespan,
)

# FastAPI exposes ``app.state`` specifically for application-scoped runtime
# data. Initializing the flag here also makes the state deterministic when a
# test client is created without entering the lifespan context.
app.state.db_ready = False


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
async def root(request: Request) -> Response:
    """Home page with portfolio showcase — requires Basic Auth password"""
    # Check Basic Auth
    user = check_basic_auth(request)
    if user is None:
        return require_auth_response()

    host = request.url.hostname or "localhost"
    port = request.url.port
    # Trust X-Forwarded-Proto from nginx reverse proxy
    scheme = request.headers.get("x-forwarded-proto", request.url.scheme) or "http"
    if port and port not in (80, 443):
        base_url = f"{scheme}://{host}:{port}"
    else:
        base_url = f"{scheme}://{host}"
    llm_model = get_llm_config()["model"]
    return HTMLResponse(
        content=build_home_page(base_url, llm_model),
        headers={"Cache-Control": "no-store"},
    )


@app.get("/health", response_model=HealthResponse, tags=["Monitoring"])
async def health_check() -> HealthResponse:
    """Report component readiness without making a paid LLM request."""

    llm_config = get_llm_config()
    llm_ready = bool(llm_config["api_key"]) and len(llm_config["api_key"]) > 10

    try:
        database_ready = await asyncio.wait_for(
            check_database_health(),
            timeout=HEALTH_CHECK_TIMEOUT_SECONDS,
        )
    except asyncio.TimeoutError:
        database_ready = False

    # Keep audit-log readiness aligned with the live database probe. This also
    # allows the application to recover automatically when PostgreSQL becomes
    # available after a degraded startup.
    app.state.db_ready = database_ready

    cache_enabled, model_info = await asyncio.gather(
        asyncio.to_thread(rag_cache.is_healthy),
        asyncio.to_thread(get_model_info),
    )
    ml_model_ready = bool(model_info.get("model_available"))

    required_components_ready = all(
        (llm_ready, database_ready, ml_model_ready)
    )

    return HealthResponse(
        status="healthy" if required_components_ready else "degraded",
        timestamp=datetime.now(timezone.utc).isoformat(),
        version=API_VERSION,
        service_name=SERVICE_NAME,
        llm_ready=llm_ready,
        cache_enabled=cache_enabled,
        database_ready=database_ready,
        ml_model_ready=ml_model_ready,
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
        # ML prediction and vector retrieval are synchronous operations. Running
        # preparation in a worker keeps concurrent HTTP requests responsive.
        scenario = await asyncio.to_thread(
            _prepare_analysis_scenario,
            request_body,
        )
        decision = await _run_analysis_with_timeout(scenario)
        grading = _grade_decision(decision)
        processing_time_ms = (time.perf_counter() - start_time) * 1000

        # Store audit log (if database is ready)
        if app.state.db_ready:
            try:
                audit_log = AuditLog(
                    session_id=str(uuid.uuid4()),
                    request_id=request_id,
                    endpoint="/analyze",
                    input_data=request_body.model_dump(),
                    output_data=decision.model_dump(),
                    model_name=get_llm_config()["model"],
                    response_time_ms=int(processing_time_ms),
                )
                session.add(audit_log)
                await session.commit()
            except Exception as e:
                await session.rollback()
                logger.warning("Audit log storage failed", error=str(e))

        await app_state.increment_request(processing_time_ms, success=True)
        logger.info(
            "Analysis completed",
            request_id=request_id,
            processing_time_ms=round(processing_time_ms, 2),
            grading_score=grading.overall_score,
        )

        return AnalysisResponse(
            request_id=request_id,
            decision=decision,
            grading=grading,
            processing_time_ms=processing_time_ms,
            timestamp=datetime.now(timezone.utc).isoformat(),
        )

    except asyncio.TimeoutError:
        processing_time_ms = (time.perf_counter() - start_time) * 1000
        await app_state.increment_request(processing_time_ms, success=False)
        logger.error(
            "Analysis timed out",
            request_id=request_id,
            timeout_seconds=ANALYSIS_TIMEOUT_SECONDS,
        )
        raise HTTPException(
            status_code=504,
            detail="AI analysis exceeded the configured time limit",
        )

    except LLMError as e:
        processing_time_ms = (time.perf_counter() - start_time) * 1000
        await app_state.increment_request(processing_time_ms, success=False)
        logger.error("LLM service unavailable", request_id=request_id, error=str(e))
        raise HTTPException(
            status_code=503,
            detail=f"AI model unavailable: {str(e)}. Check LLM provider configuration.",
        )

    except Exception as e:
        processing_time_ms = (time.perf_counter() - start_time) * 1000
        await app_state.increment_request(processing_time_ms, success=False)
        logger.error("Analysis failed", request_id=request_id, error=str(e))
        raise HTTPException(status_code=500, detail=f"Analysis failed: {str(e)}")


@app.post("/batch-analyze", tags=["Analysis"])
async def batch_analyze(
    requests: List[AnalysisRequest],
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
        for index, req in enumerate(requests):
            try:
                scenario = await asyncio.to_thread(
                    _prepare_analysis_scenario,
                    req,
                )
                decision = await _run_analysis_with_timeout(scenario)
                results.append(decision.model_dump())
                successful += 1

            except asyncio.TimeoutError:
                logger.warning("Batch item timed out", item_index=index)
                results.append({
                    "error": "AI analysis exceeded the configured time limit",
                    "error_type": "timeout",
                })
            except Exception as e:
                logger.warning(
                    "Batch item failed",
                    item_index=index,
                    error=str(e),
                )
                results.append({
                    "error": str(e),
                    "error_type": type(e).__name__,
                })

        processing_time_ms = (time.perf_counter() - start_time) * 1000
        all_succeeded = successful == len(requests)
        await app_state.increment_request(
            processing_time_ms,
            success=all_succeeded,
        )

        logger.info(
            "Batch analysis completed",
            request_id=request_id,
            total=len(requests),
            successful=successful,
            processing_time_ms=round(processing_time_ms, 2),
        )

        return {
            "request_id": request_id,
            "status": "completed" if all_succeeded else "completed_with_errors",
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
        return context if context else DEFAULT_RAG_CONTEXT
    except Exception as exc:
        # RAG is an enrichment layer. Analysis remains available with an
        # explicit neutral context, while the failure stays observable in logs.
        logger.warning("RAG context retrieval failed", error=str(exc))
        return DEFAULT_RAG_CONTEXT


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
    prediction = await asyncio.to_thread(
        predict_delivery_days,
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
        "model_info": await asyncio.to_thread(get_model_info),
    }


# ===== DEMO ENDPOINT (public, predefined scenarios only) =====

# Demo scenario parameters (without predicted_days — ML model will calculate it)
DEMO_SCENARIO_PARAMS = {
    "high": {
        "promised_days": 7.0, "distance_km": 2800, "weight_g": 4500,
        "freight_value": 65, "payment_lag_days": 5, "is_weekend_order": 1,
        "purchase_month": 11,
    },
    "moderate": {
        "promised_days": 6.0, "distance_km": 650, "weight_g": 2000,
        "freight_value": 40, "payment_lag_days": 3, "is_weekend_order": 0,
        "purchase_month": 9,
    },
    "low": {
        "promised_days": 3.0, "distance_km": 45, "weight_g": 300,
        "freight_value": 15, "payment_lag_days": 0, "is_weekend_order": 0,
        "purchase_month": 10,
    },
}

# Public demo inputs are immutable, so caching the completed response is safe
# and makes the Redis benefit visible without changing the authenticated API.
DEMO_RESPONSE_CACHE_VERSION = "v1"


class DemoRequest(BaseModel):
    """Demo analysis request — only accepts predefined scenario names"""
    scenario: str = Field(description="Scenario name: high, moderate, or low")


@app.post("/demo/analyze", tags=["Demo"])
async def demo_analyze(request_body: DemoRequest) -> Dict[str, Any]:
    """Public demo endpoint — runs predefined scenarios without API key.
    
    Uses ML model (XGBoost) to predict delivery time, then feeds into multi-agent LLM system.
    Only accepts: high, moderate, low. No custom payloads allowed.
    """
    scenario_name = request_body.scenario.lower().strip()
    if scenario_name not in DEMO_SCENARIO_PARAMS:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid scenario. Choose: high, moderate, or low.",
        )

    start_time = time.perf_counter()
    params = DEMO_SCENARIO_PARAMS[scenario_name]
    demo_cache_key = rag_cache.make_key(
        json.dumps(
            {
                "namespace": "portfolio-demo-response",
                "version": DEMO_RESPONSE_CACHE_VERSION,
                "scenario": scenario_name,
                "params": params,
                "model": get_llm_config()["model"],
            },
            sort_keys=True,
        )
    )

    cached_response = await asyncio.to_thread(rag_cache.get, demo_cache_key)
    if cached_response:
        try:
            cached_payload = json.loads(cached_response)
        except json.JSONDecodeError:
            cached_payload = None
        if isinstance(cached_payload, dict):
            return {
                "request_id": str(uuid.uuid4()),
                **cached_payload,
                "cache_enabled": True,
                "cache_hit": True,
                "processing_time_ms": round(
                    (time.perf_counter() - start_time) * 1000,
                    2,
                ),
                "timestamp": datetime.now(timezone.utc).isoformat(),
            }

    try:
        # The public demo uses fixed inputs, but follows the same ML -> RAG ->
        # agents sequence as the authenticated endpoint.
        ml_prediction = await asyncio.to_thread(
            predict_delivery_days,
            distance_km=params["distance_km"],
            weight_g=params["weight_g"],
            freight_value=params["freight_value"],
            payment_lag_days=params["payment_lag_days"],
            is_weekend_order=params["is_weekend_order"],
            purchase_month=params.get("purchase_month", 6),
        )
        ml_used = ml_prediction is not None
        predicted_days = (
            ml_prediction
            if ml_used
            else params["promised_days"] + 2.0
        )

        rag_context = await asyncio.to_thread(
            _get_rag_context,
            distance_km=params["distance_km"],
            weight_g=params["weight_g"],
            payment_lag_days=params["payment_lag_days"],
            is_weekend_order=params["is_weekend_order"],
            predicted_days=predicted_days,
            promised_days=params["promised_days"],
        )

        scenario = DeliveryScenario(
            predicted_days=predicted_days,
            promised_days=params["promised_days"],
            distance_km=params["distance_km"],
            weight_g=params["weight_g"],
            payment_lag_days=params["payment_lag_days"],
            is_weekend_order=params["is_weekend_order"],
            freight_value=params["freight_value"],
            rag_context=rag_context,
        )

        decision = await _run_analysis_with_timeout(scenario)
        grading = _grade_decision(decision)
        processing_time_ms = (time.perf_counter() - start_time) * 1000

        response_payload = {
            "scenario": scenario_name,
            "ml_prediction": {
                "predicted_days": predicted_days,
                "ml_model_used": ml_used,
                "model_type": "XGBoost Regressor (Olist dataset)",
            },
            "decision": decision.model_dump(),
            "grading": grading.model_dump(),
            "fallback_used": False,
            "fallback_reason": None,
        }
        await asyncio.to_thread(
            rag_cache.set,
            demo_cache_key,
            json.dumps(response_payload),
            600,
        )
        return {
            "request_id": str(uuid.uuid4()),
            **response_payload,
            "cache_enabled": rag_cache.is_healthy(),
            "cache_hit": False,
            "processing_time_ms": round(processing_time_ms, 2),
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }

    except asyncio.TimeoutError:
        raise HTTPException(
            status_code=504,
            detail="AI analysis exceeded the configured time limit",
        )
    except LLMError as e:
        decision = build_deterministic_fallback_decision(scenario)
        grading = _grade_decision(decision)
        processing_time_ms = (time.perf_counter() - start_time) * 1000
        logger.warning(
            "Demo used deterministic fallback",
            scenario=scenario_name,
            error=str(e),
        )
        return {
            "request_id": str(uuid.uuid4()),
            "scenario": scenario_name,
            "ml_prediction": {
                "predicted_days": predicted_days,
                "ml_model_used": ml_used,
                "model_type": "XGBoost Regressor (Olist dataset)",
            },
            "decision": decision.model_dump(),
            "grading": grading.model_dump(),
            "cache_enabled": rag_cache.is_healthy(),
            "cache_hit": False,
            "fallback_used": True,
            "fallback_reason": "LLM provider temporarily unavailable",
            "processing_time_ms": round(processing_time_ms, 2),
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Demo analysis failed: {str(e)}")


@app.get("/debug/llm-test", tags=["Monitoring"])
async def debug_llm_test(
    _: str = Depends(verify_api_key),
) -> Dict[str, Any]:
    """Run an authenticated LLM probe without exposing credential metadata."""
    config = get_llm_config()
    token = config["api_key"]
    model = config["model"]
    base_url = config["base_url"]

    diagnostics = {
        "token_present": bool(token),
        "model": model,
        "base_url": base_url,
        "llm_call_result": None,
        "error": None,
    }

    if not token or len(token) < 10:
        diagnostics["error"] = "LLM_API_KEY is missing or invalid."
        return diagnostics

    try:
        def run_probe() -> str:
            client = _get_llm_client()
            response = client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": "You are a test assistant."},
                    {"role": "user", "content": "Reply with exactly: OK"},
                ],
                temperature=0.0,
                max_tokens=10,
            )
            return response.choices[0].message.content or ""

        result = await asyncio.to_thread(run_probe)
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
