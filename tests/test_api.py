"""
Integration tests for the FastAPI endpoints.
Uses mocked LLM to test the full request/response cycle.
"""
import json
import asyncio
import pytest
from unittest.mock import AsyncMock, patch, MagicMock
from fastapi.testclient import TestClient


# Mock database before importing app
import sys
import types

mock_database = types.ModuleType("database")


async def mock_get_session():
    yield None


async def mock_init_db():
    pass


async def mock_check_database_health():
    return True


mock_database.get_session = mock_get_session
mock_database.init_db = mock_init_db
mock_database.check_database_health = mock_check_database_health
sys.modules["database"] = mock_database

from main import app, API_KEY as CONFIGURED_API_KEY  # noqa: E402
from pydantic_agents import (  # noqa: E402
    CARRIER_AGENT_PROMPT,
    LLMError,
    ORCHESTRATOR_PROMPT,
    RECOVERY_AGENT_PROMPT,
    RISK_AGENT_PROMPT,
)

client = TestClient(app)

API_KEY = CONFIGURED_API_KEY


@pytest.fixture(autouse=True)
def reset_database_readiness():
    """Keep endpoint tests independent from health-check side effects."""
    app.state.db_ready = False
    yield


# Sample LLM responses for mocking
MOCK_RISK_RESPONSE = json.dumps({
    "risk_level": "HIGH",
    "risk_score": 75,
    "primary_risk_factors": ["Long distance (2800km)", "Heavy weight (4500g)", "Weekend order"],
    "mitigation_priority": "URGENT",
    "analysis": "High risk due to 2800km distance exceeding premium threshold. Weight 4500g requires special handling. Weekend order adds 2-3 days delay. Immediate carrier upgrade recommended to meet delivery promise."
})

MOCK_CARRIER_RESPONSE = json.dumps({
    "recommended_carrier": "Premium Express",
    "current_carrier": "Standard Shipping",
    "should_upgrade": True,
    "upgrade_rationale": "Distance 2800km requires premium carrier for timely delivery.",
    "cost_impact": 50,
    "roi_analysis": "Upgrade cost R$50 prevents R$200 penalty. ROI = 300%."
})

MOCK_RECOVERY_RESPONSE = json.dumps({
    "voucher_code": "DELAY25",
    "discount_percentage": 25,
    "communication_template": "Subject: Delivery Update | We apologize for the expected delay and offer voucher DELAY25 for 25% off.",
    "timing": "Day 1: Proactive notification with voucher",
    "retention_probability": 80
})

MOCK_ORCHESTRATOR_RESPONSE = json.dumps({
    "executive_summary": "High-risk scenario requiring Premium Express upgrade and DELAY25 voucher. Confidence 85%.",
    "estimated_delivery_time": 8.0,
    "confidence_score": 85
})


def route_mock_llm(system_prompt, user_prompt, **kwargs):
    """Return agent-specific fixtures without relying on thread scheduling."""
    responses = {
        RISK_AGENT_PROMPT: MOCK_RISK_RESPONSE,
        CARRIER_AGENT_PROMPT: MOCK_CARRIER_RESPONSE,
        RECOVERY_AGENT_PROMPT: MOCK_RECOVERY_RESPONSE,
        ORCHESTRATOR_PROMPT: MOCK_ORCHESTRATOR_RESPONSE,
    }
    if system_prompt not in responses:
        raise AssertionError("Unexpected system prompt")
    return responses[system_prompt]


class TestHealthEndpoint:
    """Test health and monitoring endpoints."""

    def test_database_readiness_is_kept_in_fastapi_state(self):
        """Runtime readiness belongs to FastAPI's supported state container."""
        assert app.state.db_ready is False

    def test_health_returns_200(self):
        response = client.get("/health")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] in ("healthy", "degraded")
        assert "timestamp" in data
        assert "database_ready" in data
        assert "ml_model_ready" in data
        assert "cache_enabled" in data

    def test_status_returns_metrics(self):
        response = client.get("/status")
        assert response.status_code == 200
        data = response.json()
        assert "uptime_seconds" in data
        assert "request_count" in data


class TestAnalyzeEndpoint:
    """Test /analyze endpoint with mocked LLM."""

    @patch("main._get_rag_context", return_value="Test logistics context")
    @patch("pydantic_agents.call_ollama")
    def test_analyze_returns_full_decision(self, mock_llm, mock_rag):
        """Full analysis should return decision + grading."""
        mock_llm.side_effect = route_mock_llm

        response = client.post(
            "/analyze",
            json={
                "predicted_days": 12.5,
                "promised_days": 7,
                "distance_km": 2800,
                "weight_g": 4500,
                "freight_value": 150,
            },
            headers={"x-api-key": API_KEY},
        )

        assert response.status_code == 200
        data = response.json()

        # Check structure
        assert "decision" in data
        assert "grading" in data
        assert "processing_time_ms" in data

        # Check decision content
        decision = data["decision"]
        assert decision["risk_assessment"]["risk_level"] == "MODERATE"
        assert decision["risk_assessment"]["risk_score"] == 60
        assert "Weekend order" not in " ".join(
            decision["risk_assessment"]["primary_risk_factors"]
        )
        assert decision["carrier_recommendation"]["should_upgrade"] is True
        assert decision["recovery_plan"]["voucher_code"] == "DELAY50"

        # Check grading
        grading = data["grading"]
        assert grading["overall_score"] > 0
        assert grading["quality_level"] in ["Excellent", "Good", "Fair", "Poor"]

    def test_analyze_requires_api_key(self):
        """Should return 401/403 without API key."""
        response = client.post(
            "/analyze",
            json={"predicted_days": 8, "distance_km": 400, "weight_g": 1000, "freight_value": 30},
        )
        assert response.status_code in (401, 403)

    def test_analyze_validates_input(self):
        """Should return 422 on missing required fields."""
        response = client.post(
            "/analyze",
            json={"predicted_days": 8},
            headers={"x-api-key": API_KEY},
        )
        assert response.status_code == 422

    @patch("main._prepare_analysis_scenario")
    @patch("main._run_analysis_with_timeout", new_callable=AsyncMock)
    def test_analyze_returns_gateway_timeout(
        self,
        mock_run_analysis,
        mock_prepare_scenario,
    ):
        mock_prepare_scenario.return_value = MagicMock()
        mock_run_analysis.side_effect = asyncio.TimeoutError

        response = client.post(
            "/analyze",
            json={
                "distance_km": 400,
                "weight_g": 1000,
                "freight_value": 30,
            },
            headers={"x-api-key": API_KEY},
        )

        assert response.status_code == 504

    @patch("main._prepare_analysis_scenario")
    @patch("main._run_analysis_with_timeout", new_callable=AsyncMock)
    def test_analyze_reports_llm_failure_without_fallback(
        self,
        mock_run_analysis,
        mock_prepare_scenario,
    ):
        """Authenticated API calls must expose provider unavailability."""
        mock_prepare_scenario.return_value = MagicMock()
        mock_run_analysis.side_effect = LLMError("Daily quota exhausted")

        response = client.post(
            "/analyze",
            json={
                "distance_km": 400,
                "weight_g": 1000,
                "freight_value": 30,
            },
            headers={"x-api-key": API_KEY},
        )

        assert response.status_code == 503
        assert "Daily quota exhausted" in response.json()["detail"]


class TestBatchAnalyzeEndpoint:
    """Verify batch processing applies the same ML and RAG preparation."""

    @patch("main._get_rag_context", return_value="Retrieved carrier rules")
    @patch("main.predict_delivery_days", return_value=6.2)
    @patch("main.run_multi_agent_analysis_parallel")
    def test_batch_resolves_missing_predicted_days(
        self,
        mock_workflow,
        mock_predict,
        mock_rag,
    ):
        mock_decision = MagicMock()
        mock_decision.model_dump.return_value = {"status": "ok"}
        mock_workflow.return_value = mock_decision

        response = client.post(
            "/batch-analyze",
            json=[{
                "distance_km": 450,
                "weight_g": 1200,
                "freight_value": 45,
            }],
            headers={"x-api-key": API_KEY},
        )

        assert response.status_code == 200
        assert response.json()["successful"] == 1
        scenario = mock_workflow.call_args.args[0]
        assert scenario.predicted_days == 6.2
        assert scenario.rag_context == "Retrieved carrier rules"
        mock_predict.assert_called_once()
        mock_rag.assert_called_once()


class TestDemoEndpoint:
    """Test /demo/analyze public endpoint."""

    @patch("main._run_analysis_with_timeout")
    @patch("main.rag_cache.get")
    def test_demo_returns_cached_complete_response(
        self,
        mock_cache_get,
        mock_workflow,
    ):
        """A repeated fixed scenario should skip ML, RAG, and all agents."""
        mock_cache_get.return_value = json.dumps(
            {
                "scenario": "moderate",
                "ml_prediction": {"predicted_days": 1.5},
                "decision": {"confidence_score": 80},
                "grading": {"overall_score": 82.3},
                "fallback_used": False,
                "fallback_reason": None,
            }
        )

        response = client.post("/demo/analyze", json={"scenario": "moderate"})

        assert response.status_code == 200
        assert response.json()["cache_hit"] is True
        assert response.json()["cache_enabled"] is True
        mock_workflow.assert_not_called()

    @patch("main._get_rag_context", return_value="Test logistics context")
    @patch("pydantic_agents.call_ollama")
    def test_demo_accepts_predefined_scenarios(self, mock_llm, mock_rag):
        """Demo endpoint should work without API key."""
        mock_llm.side_effect = route_mock_llm

        response = client.post("/demo/analyze", json={"scenario": "high"})
        assert response.status_code == 200
        data = response.json()
        assert data["scenario"] == "high"
        assert "decision" in data
        assert "grading" in data
        assert data["fallback_used"] is False
        assert "cache_enabled" in data
        assert data["cache_hit"] is False

    @patch("main._get_rag_context", return_value="Test logistics context")
    @patch("pydantic_agents.call_ollama")
    def test_demo_uses_typed_fallback_when_llm_is_unavailable(
        self,
        mock_llm,
        mock_rag,
    ):
        """The public demo remains usable and labels deterministic output."""
        mock_llm.side_effect = LLMError("Daily quota exhausted")

        response = client.post("/demo/analyze", json={"scenario": "high"})

        assert response.status_code == 200
        data = response.json()
        carrier = data["decision"]["carrier_recommendation"]
        assert data["fallback_used"] is True
        assert "cache_enabled" in data
        assert data["cache_hit"] is False
        assert data["fallback_reason"] == (
            "LLM provider temporarily unavailable"
        )
        assert carrier["quote_source"] == "portfolio_rate_card_v1"
        assert carrier["estimated_cost"] > 0
        assert data["decision"]["confidence_score"] == 65.0

    def test_demo_rejects_invalid_scenario(self):
        """Should reject unknown scenario names."""
        response = client.post("/demo/analyze", json={"scenario": "extreme"})
        assert response.status_code == 400


class TestPortfolioAuth:
    """Test Basic Auth on portfolio page."""

    def test_root_requires_auth(self):
        """Root page should return 401 without credentials."""
        response = client.get("/", follow_redirects=False)
        assert response.status_code == 401
        assert "WWW-Authenticate" in response.headers

    def test_root_accepts_valid_password(self):
        """Root page should return 200 with correct password."""
        import base64
        creds = base64.b64encode(b"user:portfolio2026").decode()
        response = client.get("/", headers={"Authorization": f"Basic {creds}"})
        assert response.status_code == 200
        assert "Multi-Agent" in response.text
        assert response.headers["Cache-Control"] == "no-store"
        assert "Azure VM + Docker Compose + GitHub Actions" in response.text
        assert "DigitalOcean" not in response.text
        assert "Rate Limiting" not in response.text
        assert "Few-shot" not in response.text
        assert 'id="metric-processing"' in response.text


class TestDebugEndpointAuth:
    """Diagnostic endpoints must not expose model details anonymously."""

    def test_llm_debug_requires_api_key(self):
        response = client.get("/debug/llm-test")
        assert response.status_code in (401, 403)
