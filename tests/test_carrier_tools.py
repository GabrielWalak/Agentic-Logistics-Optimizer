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
            "roi_analysis": "The verified standard quote requires no upgrade.",
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
