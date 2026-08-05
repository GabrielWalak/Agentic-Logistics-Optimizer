"""Unit tests for deterministic, typed carrier quote tools."""

import json

import pytest
from pydantic import ValidationError

import pydantic_agents as agents

from carrier_tools import (
    CarrierQuoteInput,
    get_all_carrier_quotes,
    get_carrier_quote,
    select_carrier_quote,
)


def test_quote_is_deterministic_and_typed():
    tool_input = CarrierQuoteInput(
        carrier="Premium Express",
        distance_km=800,
        weight_g=1500,
    )

    first = get_carrier_quote(tool_input)
    second = get_carrier_quote(tool_input)

    assert first == second
    assert first.available is True
    assert first.estimated_cost > 0
    assert first.estimated_transit_days > 0
    assert first.source == "portfolio_rate_card_v1"


def test_tool_rejects_invalid_operational_input():
    with pytest.raises(ValidationError):
        CarrierQuoteInput(
            carrier="Premium Express",
            distance_km=-10,
            weight_g=1000,
        )


def test_unavailable_llm_choice_uses_fastest_upgrade():
    quotes = get_all_carrier_quotes(
        distance_km=2800,
        weight_g=4500,
    )

    selected = select_carrier_quote(
        quotes=quotes,
        requested_carrier="Standard Shipping",
        should_upgrade=True,
    )

    assert selected.available is True
    assert selected.carrier == "Premium Express"


def test_carrier_agent_does_not_trust_generated_price(monkeypatch):
    """The LLM may explain a choice but cannot override verified quote data."""

    def fake_llm(*args, **kwargs):
        return json.dumps({
            "recommended_carrier": "Standard Shipping",
            "current_carrier": "Standard Shipping",
            "should_upgrade": False,
            "upgrade_rationale": "The standard option meets the promise.",
            "cost_impact": 999_999,
            "roi_analysis": "Assumed penalty R$200 produces a 300% ROI.",
        })

    monkeypatch.setattr(agents, "call_ollama", fake_llm)
    scenario = agents.DeliveryScenario(
        predicted_days=5,
        promised_days=7,
        distance_km=400,
        weight_g=1000,
        payment_lag_days=1,
        is_weekend_order=0,
        freight_value=25,
        rag_context="Standard regional delivery rules.",
    )

    result = agents.run_carrier_optimization(
        scenario,
        {"risk_level": "LOW", "risk_score": 30},
    )

    assert result["recommended_carrier"] == "Standard Shipping"
    assert result["cost_impact"] == 0
    assert result["cost_impact"] != 999_999
    assert result["quote_source"] == "portfolio_rate_card_v1"
    assert "300%" not in result["roi_analysis"]


def test_deterministic_fallback_avoids_unnecessary_upgrade():
    """An on-time, low-risk shipment should keep standard delivery."""
    scenario = agents.DeliveryScenario(
        predicted_days=3,
        promised_days=5,
        distance_km=100,
        weight_g=500,
        payment_lag_days=1,
        is_weekend_order=0,
        freight_value=20,
        rag_context="Standard regional delivery rules.",
    )

    decision = agents.build_deterministic_fallback_decision(scenario)

    assert decision.risk_assessment.risk_level == "MINIMAL"
    assert decision.carrier_recommendation.recommended_carrier == (
        "Standard Shipping"
    )
    assert decision.carrier_recommendation.should_upgrade is False
    assert decision.carrier_recommendation.cost_impact == 0


def test_recovery_agent_enforces_voucher_policy(monkeypatch):
    """An LLM cannot issue a voucher when no delay is predicted."""

    def fake_llm(*args, **kwargs):
        return json.dumps({
            "voucher_code": "EXPRESS_FREE",
            "discount_percentage": 50,
            "communication_template": "Use an unsupported recovery benefit.",
            "timing": "Immediately",
            "retention_probability": 82,
        })

    monkeypatch.setattr(agents, "call_ollama", fake_llm)
    scenario = agents.DeliveryScenario(
        predicted_days=6.9,
        promised_days=7,
        distance_km=2800,
        weight_g=4500,
        payment_lag_days=5,
        is_weekend_order=1,
        freight_value=65,
        rag_context="Standard regional delivery rules.",
    )

    result = agents.run_recovery_strategy(
        scenario,
        {"risk_level": "HIGH", "risk_score": 78},
    )

    assert result["voucher_code"] is None
    assert result["discount_percentage"] == 0
    assert "on track" in result["communication_template"]


def test_risk_agent_enforces_score_level_and_grounded_narrative(monkeypatch):
    """Generated scoring and unsupported statistics cannot reach the API."""

    def fake_llm(*args, **kwargs):
        return json.dumps({
            "risk_level": "CRITICAL",
            "risk_score": 80,
            "primary_risk_factors": ["Unsupported remote-region risk"],
            "mitigation_priority": "URGENT",
            "analysis": (
                "This is a critical risk with average delivery of 15-30 days "
                "and a 98% failure probability."
            ),
        })

    monkeypatch.setattr(agents, "call_ollama", fake_llm)
    scenario = agents.DeliveryScenario(
        predicted_days=6.9,
        promised_days=7,
        distance_km=2800,
        weight_g=4500,
        payment_lag_days=5,
        is_weekend_order=1,
        freight_value=65,
        rag_context="General long-haul guidance.",
    )

    result = agents.run_risk_assessment(scenario)

    assert result["risk_score"] == 40
    assert result["risk_level"] == "LOW"
    assert result["mitigation_priority"] == "MEDIUM"
    assert "15-30" not in result["analysis"]
    assert "98%" not in result["analysis"]
    assert "40/100" in result["analysis"]
