"""
Tests for the Response Grading Framework.
Validates that the grader correctly scores agent responses
based on logical consistency, not just JSON format.
"""
import json
import pytest
from prompt_engineering import ResponseGrader


class TestRiskAssessmentGrading:
    """Test risk assessment grading logic."""

    def test_perfect_high_risk_response(self):
        """A well-formed HIGH risk response should score 85+."""
        response = json.dumps({
            "risk_level": "HIGH",
            "risk_score": 75,
            "primary_risk_factors": [
                "Long distance (2800km)",
                "Heavy weight (4500g)",
                "Payment lag (5 days)"
            ],
            "mitigation_priority": "URGENT",
            "analysis": (
                "The delivery scenario presents significant risk due to the 2800km distance "
                "which exceeds the 2000km threshold for premium carrier requirement. "
                "Combined with 4500g weight requiring special handling and a 5-day payment lag, "
                "the composite risk score of 75 places this firmly in the HIGH category. "
                "Immediate carrier upgrade and proactive customer communication are recommended."
            )
        })
        score, details = ResponseGrader.grade_risk_assessment(response)
        assert score >= 85, f"Expected 85+, got {score}. Details: {details}"
        assert details["structure"] is True
        assert details["risk_level_valid"] is True
        assert details["score_alignment"] is True

    def test_score_level_mismatch_penalized(self):
        """Score 45 labeled as HIGH should be penalized (HIGH = 61-80)."""
        response = json.dumps({
            "risk_level": "HIGH",
            "risk_score": 45,
            "primary_risk_factors": ["Distance (500km)", "Weight (2000g)"],
            "mitigation_priority": "MEDIUM",
            "analysis": "Moderate risk scenario with some concerns about delivery timing."
        })
        score, details = ResponseGrader.grade_risk_assessment(response)
        assert details["score_alignment"] is False
        assert score < 70, f"Misaligned score should be penalized, got {score}"

    def test_generic_factors_penalized(self):
        """Vague factors without measurements should lose points."""
        response = json.dumps({
            "risk_level": "MODERATE",
            "risk_score": 55,
            "primary_risk_factors": ["Risk factor 1", "Risk factor 2", "General concern"],
            "mitigation_priority": "MEDIUM",
            "analysis": "There are some risks that need to be addressed in this scenario."
        })
        score, details = ResponseGrader.grade_risk_assessment(response)
        assert details["factors_specificity"] is False

    def test_invalid_json_returns_zero(self):
        """Invalid JSON should score 0."""
        score, details = ResponseGrader.grade_risk_assessment("not valid json {{{")
        assert score == 0
        assert "error" in details

    def test_missing_fields_returns_low_score(self):
        """Missing required fields should score very low."""
        response = json.dumps({"risk_level": "HIGH"})
        score, details = ResponseGrader.grade_risk_assessment(response)
        assert score <= 20


class TestCarrierGrading:
    """Test carrier recommendation grading logic."""

    def test_upgrade_true_with_zero_cost_penalized(self):
        """Logic error: upgrade=true but cost_impact=0 should be penalized."""
        response = json.dumps({
            "recommended_carrier": "Premium Express",
            "should_upgrade": True,
            "cost_impact": 0,
            "roi_analysis": "No additional cost needed"
        })
        score, details = ResponseGrader.grade_carrier_recommendation(response)
        assert "Logic error" in str(details.get("deductions", []))

    def test_valid_carrier_with_roi(self):
        """Valid carrier with proper ROI analysis should score high."""
        response = json.dumps({
            "recommended_carrier": "Premium Express",
            "should_upgrade": True,
            "cost_impact": 50,
            "roi_analysis": "Upgrade cost R$50 saves R$200 in penalties. ROI = 300%."
        })
        score, details = ResponseGrader.grade_carrier_recommendation(response)
        assert score >= 80, f"Expected 80+, got {score}"
        assert details.get("roi_valid") is True


class TestRecoveryGrading:
    """Test recovery plan grading logic."""

    def test_voucher_discount_alignment(self):
        """DELAY25 should have ~25% discount."""
        response = json.dumps({
            "voucher_code": "DELAY25",
            "discount_percentage": 25,
            "communication_template": "Subject: Delivery Update | We apologize for the delay and offer you code DELAY25.",
            "timing": "Day 1: Proactive notification",
            "retention_probability": 80
        })
        score, details = ResponseGrader.grade_recovery_plan(response)
        assert details.get("discount_valid") is True
        assert details.get("voucher_valid") is True
        assert score >= 80

    def test_misaligned_voucher_discount(self):
        """DELAY15 with 50% discount should be flagged."""
        response = json.dumps({
            "voucher_code": "DELAY15",
            "discount_percentage": 50,
            "communication_template": "We offer a discount",
            "timing": "Day 1",
            "retention_probability": 70
        })
        score, details = ResponseGrader.grade_recovery_plan(response)
        assert details.get("discount_valid") is not True
