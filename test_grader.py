#!/usr/bin/env python3
"""
Test improved ResponseGrader
Validates that good/bad responses receive different scores
"""
from prompt_engineering import ResponseGrader

grader = ResponseGrader()

print("=" * 80)
print(" TESTING IMPROVED RESPONSE GRADER")
print("=" * 80)

# Test 1: Good response (aligned score with level)
print("\n📝 TEST 1: GOOD RESPONSE (aligned score-level)")
print("-" * 80)
good_risk = """{
  "risk_level": "HIGH",
  "risk_score": 75,
  "primary_risk_factors": ["Distance 2800km", "Weight 4500g", "Payment lag 5 days"],
  "analysis": "This is a high-risk scenario. The predicted delivery of 12.5 days exceeds the promised 7 days by 5.5 days, indicating significant delay risk. The 2800km distance requires premium logistics handling."
}"""

score, details = grader.grade_risk_assessment(good_risk)
print(f"Score: {score}/100")
print(f"Passed: {[k for k, v in details.items() if v is True]}")
print(f"Deductions: {details.get('deductions', [])}")

# Test 2: Misaligned response (score doesn't match level)
print("\n❌ TEST 2: BAD RESPONSE (misaligned score-level)")
print("-" * 80)
bad_risk = """{
  "risk_level": "HIGH",
  "risk_score": 25,
  "primary_risk_factors": ["Risk", "Factors"],
  "analysis": "Bad analysis."
}"""

score, details = grader.grade_risk_assessment(bad_risk)
print(f"Score: {score}/100")
print(f"Passed: {[k for k, v in details.items() if v is True]}")
print(f"Deductions: {details.get('deductions', [])}")

# Test 3: Vague factors
print("\n⚠️  TEST 3: VAGUE FACTORS (no specific measurements)")
print("-" * 80)
vague_risk = """{
  "risk_level": "MODERATE",
  "risk_score": 55,
  "primary_risk_factors": ["Risk factor 1", "Risk factor 2"],
  "analysis": "This delivery has some risks that need to be considered carefully and monitored."
}"""

score, details = grader.grade_risk_assessment(vague_risk)
print(f"Score: {score}/100")
print(f"Passed: {[k for k, v in details.items() if v is True]}")
print(f"Deductions: {details.get('deductions', [])}")

# Test 4: Carrier - Mismatched upgrade logic
print("\n🚚 TEST 4: CARRIER - UPGRADE=TRUE but COST=0 (logic error)")
print("-" * 80)
bad_carrier = """{
  "recommended_carrier": "Premium Express",
  "current_carrier": "Standard",
  "should_upgrade": true,
  "upgrade_rationale": "Need faster delivery",
  "cost_impact": 0,
  "roi_analysis": "No additional cost"
}"""

score, details = grader.grade_carrier_recommendation(bad_carrier)
print(f"Score: {score}/100")
print(f"Passed: {[k for k, v in details.items() if v is True]}")
print(f"Deductions: {details.get('deductions', [])}")

# Test 5: Recovery - Voucher-discount mismatch
print("\n💰 TEST 5: RECOVERY - VOUCHER-DISCOUNT MISMATCH")
print("-" * 80)
bad_recovery = """{
  "voucher_code": "DELAY25",
  "discount_percentage": 50,
  "communication_template": "Here is your voucher",
  "retention_probability": 60
}"""

score, details = grader.grade_recovery_plan(bad_recovery)
print(f"Score: {score}/100")
print(f"Passed: {[k for k, v in details.items() if v is True]}")
print(f"Deductions: {details.get('deductions', [])}")

print("\n" + "=" * 80)
print(" SUMMARY: Good responses should score 70+, bad responses <70")
print("=" * 80)
