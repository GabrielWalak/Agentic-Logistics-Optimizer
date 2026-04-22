"""
Prompt Engineering & Response Grading Framework
Optimized prompts with evaluation metrics for logistics agent system
"""
import json
import re
from typing import Dict, List, Tuple


# ===== OPTIMIZED PROMPTS (Prompt Engineering) =====

RISK_AGENT_PROMPT_V2 = """You are a Senior Risk Assessment Specialist for OLIST Logistics.

CONTEXT: Analyze delivery scenarios with structured logic using numerical scoring.

YOUR ROLE:
1. Evaluate 5 primary risk dimensions: Distance, Weight, Time Buffer, Payment Lag, Temporal Factors
2. Assign dimension scores (0-20 each, max 100 total)
3. Map composite score to risk level: 0-20=MINIMAL, 21-40=LOW, 41-60=MODERATE, 61-80=HIGH, 81-100=CRITICAL
4. Identify top 3 risk factors by impact
5. Recommend specific mitigation actions

RULES:
- Distance >1500km adds 20 points (premium carrier needed)
- Weight >3kg adds 15 points (insurance required)
- Delay >3 days adds 25 points (critical intervention needed)
- Payment lag >5 days adds 10 points
- Weekend order adds 5 points

EXPECTED OUTPUT (JSON ONLY):
{
  "risk_level": "HIGH|CRITICAL|MODERATE|LOW|MINIMAL",
  "risk_score": 75,
  "primary_risk_factors": ["Long distance (2800km)", "Heavy weight (4500g)", "Weekend order"],
  "mitigation_priority": "URGENT|HIGH|MEDIUM|LOW",
  "analysis": "Detailed explanation of risk drivers and recommended actions"
}"""


CARRIER_AGENT_PROMPT_V2 = """You are a Carrier Optimization Specialist. Your goal: minimize cost while maximizing delivery reliability.

CARRIER OPTIONS by distance/weight:
- Standard Shipping: <500km, <2kg, 5-7 days, cost=$15/unit
- Premium Express: 500-1500km, any weight, 2-3 days, cost=$45/unit
- SEDEX: 1000-2500km, <5kg, 4-6 days, cost=$35/unit
- International: >2500km, special handling, 7-15 days, cost=$60/unit

DECISION LOGIC:
- If delay_risk > 2 days: recommend upgrade to faster carrier
- If cost_impact < risk_penalty: recommend upgrade
- Otherwise: keep current carrier

EXPECTED OUTPUT (JSON ONLY):
{
  "recommended_carrier": "SEDEX|Premium Express|Standard|International",
  "current_carrier": "Standard Shipping",
  "should_upgrade": true|false,
  "upgrade_rationale": "Cost-benefit analysis or risk mitigation reason",
  "cost_impact": 30,
  "roi_analysis": "Upgrade cost R$30 saves R$150 in penalties (ROI 400%)"
}"""


RECOVERY_AGENT_PROMPT_V2 = """You are a Customer Recovery & Retention Specialist. Maximize lifetime value through targeted interventions.

VOUCHER STRATEGY by delay risk:
- DELAY15 (15% discount): 0-1 day delay, 70% retention rate
- DELAY25 (25% discount): 1-3 day delay, 80% retention rate
- DELAY50 (50% + free shipping): 3-7 day delay, 85% retention rate
- EXPRESS_FREE (free express next order): >7 days, 90% retention rate

COMMUNICATION TIMING:
- Day 1: Proactive notification (preferred)
- Day 3: Escalated communication with voucher
- Day 5: Premium recovery offer

EXPECTED OUTPUT (JSON ONLY):
{
  "voucher_code": "DELAY25|DELAY50|EXPRESS_FREE|null",
  "discount_percentage": 25,
  "communication_template": "Subject line and first 50 chars of message",
  "timing": "Day 1: Proactive notification",
  "retention_probability": 82.5
}"""


# ===== RESPONSE GRADING FRAMEWORK =====

class ResponseGrader:
    """
    Grading system for LLM agent responses
    Evaluates: Structure, Completeness, Logic, Accuracy
    """
    
    @staticmethod
    def grade_risk_assessment(response: str) -> Tuple[float, Dict]:
        """
        Grade risk assessment response with stricter logic validation
        
        Returns:
            (score 0-100, details dict)
        """
        try:
            # Parse JSON
            data = json.loads(response.replace('```json', '').replace('```', '').strip())
            score = 0
            details = {
                "structure": False,
                "risk_level_valid": False,
                "score_alignment": False,  # NEW: Score matches risk level
                "factors_specificity": False,  # NEW: Factors are specific
                "analysis_depth": False
            }
            deductions = []
            
            # Check structure (20 pts)
            required_fields = ["risk_level", "risk_score", "primary_risk_factors", "analysis"]
            if all(field in data for field in required_fields):
                score += 20
                details["structure"] = True
            else:
                deductions.append("Missing required fields")
                return score, details
            
            # Check risk_level validity (20 pts)
            valid_levels = ["MINIMAL", "LOW", "MODERATE", "HIGH", "CRITICAL"]
            risk_level = data.get("risk_level")
            if risk_level in valid_levels:
                score += 20
                details["risk_level_valid"] = True
            else:
                deductions.append(f"Invalid risk_level: {risk_level}")
            
            # NEW: Validate score aligns with risk_level (25 pts)
            risk_score = data.get("risk_score", -1)
            level_ranges = {
                "MINIMAL": (0, 20),
                "LOW": (21, 40),
                "MODERATE": (41, 60),
                "HIGH": (61, 80),
                "CRITICAL": (81, 100)
            }
            
            if risk_level in level_ranges:
                min_val, max_val = level_ranges[risk_level]
                if min_val <= risk_score <= max_val:
                    score += 25
                    details["score_alignment"] = True
                else:
                    # Mismatch penalty
                    deductions.append(
                        f"Score-level mismatch: {risk_level} should be {min_val}-{max_val}, got {risk_score}"
                    )
                    score = max(0, score - 15)  # Harsh penalty
            
            # NEW: Check factors are specific, not generic (20 pts)
            factors = data.get("primary_risk_factors", [])
            if isinstance(factors, list) and 2 <= len(factors) <= 5:
                # Check specificity: should contain numbers, times, or units
                specificity_score = 0
                for factor in factors:
                    factor_str = str(factor).lower()
                    if any(x in factor_str for x in ["km", "kg", "g", "day", "days", "%", "lag", "distance", "weight"]):
                        specificity_score += 1
                
                if specificity_score >= len(factors) * 0.6:  # 60% must be specific
                    score += 20
                    details["factors_specificity"] = True
                else:
                    deductions.append("Factors lack specificity (need measurements/numbers)")
                    score = max(0, score - 10)
            else:
                deductions.append(f"Invalid factors list: {len(factors)} items")
            
            # NEW: Check analysis depth (15 pts)
            analysis = data.get("analysis", "")
            if isinstance(analysis, str):
                analysis_words = len(analysis.split())
                sentences = analysis.count(".") + analysis.count("!") + analysis.count("?")
                
                if analysis_words > 80 and sentences >= 2:  # Meaningful depth
                    score += 15
                    details["analysis_depth"] = True
                elif analysis_words > 50:
                    score += 8  # Partial credit
                    deductions.append("Analysis is too brief")
                else:
                    deductions.append(f"Analysis insufficient: {analysis_words} words")
            
            details["deductions"] = deductions
            return min(score, 100), details
            
        except json.JSONDecodeError:
            return 0, {"error": "Invalid JSON format"}
    
    
    @staticmethod
    def grade_carrier_recommendation(response: str) -> Tuple[float, Dict]:
        """Grade carrier optimization response with logic validation"""
        try:
            data = json.loads(response.replace('```json', '').replace('```', '').strip())
            score = 0
            details = {}
            deductions = []
            
            # Check structure (20 pts)
            required = ["recommended_carrier", "should_upgrade", "cost_impact", "roi_analysis"]
            if all(f in data for f in required):
                score += 20
                details["structure"] = True
            else:
                return 0, {"error": "Missing required fields", "deductions": deductions}
            
            # Check carrier valid (20 pts)
            valid_carriers = ["Standard", "Premium Express", "SEDEX", "International", "Regional"]
            carrier = str(data.get("recommended_carrier", ""))
            if any(c in carrier for c in valid_carriers):
                score += 20
                details["carrier_valid"] = True
            else:
                deductions.append(f"Invalid carrier: {carrier}")
            
            # Check upgrade is boolean (15 pts)
            if isinstance(data.get("should_upgrade"), bool):
                score += 15
                details["upgrade_bool"] = True
            else:
                deductions.append("should_upgrade must be boolean")
            
            # Check cost_impact is number and logic (20 pts)
            cost = data.get("cost_impact")
            if isinstance(cost, (int, float)) and cost >= 0:
                # If upgrade=true, cost should be > 0
                upgrade = data.get("should_upgrade")
                if upgrade and cost == 0:
                    deductions.append("Logic error: upgrade=True but cost_impact=0")
                    score = max(0, score - 10)
                else:
                    score += 20
                    details["cost_valid"] = True
            else:
                deductions.append(f"Invalid cost_impact: {cost}")
            
            # NEW: Check ROI contains numerical analysis (25 pts)
            roi = str(data.get("roi_analysis", "")).lower()
            has_percentage = "%" in roi
            has_roi_word = "roi" in roi or "return" in roi
            has_reasoning = len(roi) > 40
            
            if has_percentage and (has_roi_word or has_reasoning):
                score += 25
                details["roi_valid"] = True
            elif has_percentage or has_reasoning:
                score += 12  # Partial
                deductions.append("ROI analysis incomplete: missing percentage or clear reasoning")
            else:
                deductions.append("ROI analysis too vague or missing numbers")
            
            details["deductions"] = deductions
            return min(score, 100), details
            
        except json.JSONDecodeError:
            return 0, {"error": "Invalid JSON"}
    
    
    @staticmethod
    def grade_recovery_plan(response: str) -> Tuple[float, Dict]:
        """Grade customer recovery response with logic validation"""
        try:
            data = json.loads(response.replace('```json', '').replace('```', '').strip())
            score = 0
            details = {}
            deductions = []
            
            # Structure (20 pts)
            required = ["voucher_code", "discount_percentage", "communication_template", "retention_probability"]
            if all(f in data for f in required):
                score += 20
                details["structure"] = True
            else:
                return 0, {"error": "Missing required fields", "deductions": deductions}
            
            # Voucher validity (20 pts)
            voucher = data.get("voucher_code")
            valid_vouchers = ["DELAY15", "DELAY25", "DELAY50", "EXPRESS_FREE", None]
            if voucher in valid_vouchers:
                score += 20
                details["voucher_valid"] = True
            else:
                deductions.append(f"Invalid voucher: {voucher}")
            
            # NEW: Discount aligns with voucher strategy (20 pts)
            discount = data.get("discount_percentage")
            voucher_discounts = {
                "DELAY15": 15,
                "DELAY25": 25,
                "DELAY50": 50,
                "EXPRESS_FREE": 100,
                None: 0
            }
            
            if isinstance(discount, (int, float)) and 0 <= discount <= 100:
                expected = voucher_discounts.get(voucher, 0)
                if discount == expected or (expected > 0 and abs(discount - expected) <= 5):
                    score += 20
                    details["discount_valid"] = True
                else:
                    deductions.append(f"Discount {discount}% doesn't align with {voucher} ({expected}% expected)")
                    score = max(0, score - 8)
            else:
                deductions.append(f"Invalid discount: {discount}")
            
            # NEW: Retention aligns with discount (20 pts)
            retention = data.get("retention_probability")
            voucher_retention = {
                "DELAY15": 70,
                "DELAY25": 80,
                "DELAY50": 85,
                "EXPRESS_FREE": 90,
                None: 50
            }
            
            if isinstance(retention, (int, float)) and 0 <= retention <= 100:
                expected_ret = voucher_retention.get(voucher, 50)
                if abs(retention - expected_ret) <= 10:  # Allow ±10 variance
                    score += 20
                    details["retention_valid"] = True
                else:
                    deductions.append(
                        f"Retention {retention}% inconsistent with {voucher} ({expected_ret}% expected)"
                    )
            else:
                deductions.append(f"Invalid retention_probability: {retention}")
            
            # NEW: Communication template specificity (20 pts)
            template = str(data.get("communication_template", ""))
            if len(template) > 30 and any(x in template.lower() for x in ["subject", "day", "voucher", "code", "discount"]):
                score += 20
                details["communication_valid"] = True
            elif len(template) > 15:
                score += 10
                deductions.append("Communication template too generic")
            else:
                deductions.append("Communication template missing or empty")
            
            details["deductions"] = deductions
            return min(score, 100), details
            
        except json.JSONDecodeError:
            return 0, {"error": "Invalid JSON"}


# ===== GRADING REPORT GENERATOR =====

def generate_grading_report(responses: Dict[str, str]) -> Dict:
    """
    Generate comprehensive grading report for all agent responses
    
    Args:
        responses: {"risk": response_str, "carrier": response_str, "recovery": response_str}
        
    Returns:
        Full grading report with scores and recommendations
    """
    grader = ResponseGrader()
    
    risk_score, risk_details = grader.grade_risk_assessment(responses.get("risk", "{}"))
    carrier_score, carrier_details = grader.grade_carrier_recommendation(responses.get("carrier", "{}"))
    recovery_score, recovery_details = grader.grade_recovery_plan(responses.get("recovery", "{}"))
    
    overall_score = (risk_score + carrier_score + recovery_score) / 3
    
    return {
        "overall_score": round(overall_score, 1),
        "quality_level": "Excellent" if overall_score >= 85 else "Good" if overall_score >= 70 else "Fair" if overall_score >= 50 else "Poor",
        "agents": {
            "risk_assessment": {"score": risk_score, "details": risk_details},
            "carrier_optimization": {"score": carrier_score, "details": carrier_details},
            "recovery_strategy": {"score": recovery_score, "details": recovery_details}
        },
        "recommendations": [
            "Improve JSON structure compliance" if risk_score < 50 else None,
            "Add more specific ROI analysis" if carrier_score < 60 else None,
            "Expand recovery communication templates" if recovery_score < 70 else None
        ]
    }


if __name__ == "__main__":
    print("Prompt Engineering & Grading Framework loaded.")
    print(f"Risk Agent Prompt V2: {len(RISK_AGENT_PROMPT_V2)} chars")
    print(f"Carrier Agent Prompt V2: {len(CARRIER_AGENT_PROMPT_V2)} chars")
    print(f"Recovery Agent Prompt V2: {len(RECOVERY_AGENT_PROMPT_V2)} chars")
    
    # Test grader with sample response
    sample_risk = '{"risk_level": "HIGH", "risk_score": 75, "primary_risk_factors": ["Long distance", "Heavy weight"], "analysis": "This is a sample analysis."}'
    score, details = ResponseGrader.grade_risk_assessment(sample_risk)
    print(f"\nSample Risk Response Score: {score}/100")
    print(f"Details: {details}")
