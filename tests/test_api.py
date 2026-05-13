"""
Integration tests for the FastAPI endpoints.
Uses mocked LLM to test the full request/response cycle.
"""
import json
import pytest
from unittest.mock import patch, MagicMock
from fastapi.testclient import TestClient


# Mock database before importing app
import sys
import types

mock_database = types.ModuleType("database")


async def mock_get_session():
    yield None


async def mock_init_db():
    pass


mock_database.get_session = mock_get_session
mock_database.init_db = mock_init_db
sys.modules["database"] = mock_database

from main import app, API_KEY as CONFIGURED_API_KEY  # noqa: E402

app.db_ready = False

client = TestClient(app)

API_KEY = CONFIGURED_API_KEY


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


class TestHealthEndpoint:
    """Test health and monitoring endpoints."""

    def test_health_returns_200(self):
        response = client.get("/health")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "healthy"
        assert "timestamp" in data

    def test_status_returns_metrics(self):
        response = client.get("/status")
        assert response.status_code == 200
        data = response.json()
        assert "uptime_seconds" in data
        assert "request_count" in data


class TestAnalyzeEndpoint:
    """Test /analyze endpoint with mocked LLM."""

    @patch("pydantic_agents.call_ollama")
    def test_analyze_returns_full_decision(self, mock_llm):
        """Full analysis should return decision + grading."""
        mock_llm.side_effect = [
            MOCK_RISK_RESPONSE,
            MOCK_CARRIER_RESPONSE,
            MOCK_RECOVERY_RESPONSE,
            MOCK_ORCHESTRATOR_RESPONSE,
        ]

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
        assert decision["risk_assessment"]["risk_level"] == "HIGH"
        assert decision["carrier_recommendation"]["should_upgrade"] is True
        assert decision["recovery_plan"]["voucher_code"] == "DELAY25"

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


class TestDemoEndpoint:
    """Test /demo/analyze public endpoint."""

    @patch("pydantic_agents.call_ollama")
    def test_demo_accepts_predefined_scenarios(self, mock_llm):
        """Demo endpoint should work without API key."""
        mock_llm.side_effect = [
            MOCK_RISK_RESPONSE,
            MOCK_CARRIER_RESPONSE,
            MOCK_RECOVERY_RESPONSE,
            MOCK_ORCHESTRATOR_RESPONSE,
        ]

        response = client.post("/demo/analyze", json={"scenario": "high"})
        assert response.status_code == 200
        data = response.json()
        assert data["scenario"] == "high"
        assert "decision" in data
        assert "grading" in data

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
