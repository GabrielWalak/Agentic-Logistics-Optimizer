"""
3 Real-World Logistics Scenarios for Testing AgenticAI Multi-Agent System
"""
from pydantic_agents import (
    DeliveryScenario,
    run_multi_agent_analysis_parallel,
    IntegratedDecision
)
import json
from typing import Dict


# ===== SCENARIO 1: HIGH-RISK LONG DISTANCE =====
SCENARIO_1_HIGH_RISK = DeliveryScenario(
    predicted_days=12.5,
    promised_days=7.0,
    distance_km=2800,
    weight_g=4500,
    payment_lag_days=5,
    is_weekend_order=1,
    freight_value=65.00,
    rag_context="""
    Distance Guidelines: Deliveries over 2000km require premium carriers.
    Weight Rules: Packages 4-5kg need reinforced packaging and insurance.
    Weekend Impact: Weekend orders incur 2-3 day penalty.
    Payment Lag: 5+ day payment lags increase risk by 35%.
    Recovery: High-risk customers respond well to proactive vouchers (25-50%).
    """
)


# ===== SCENARIO 2: LOW-RISK LOCAL DELIVERY =====
SCENARIO_2_LOW_RISK = DeliveryScenario(
    predicted_days=2.0,
    promised_days=3.0,
    distance_km=45.0,
    weight_g=300.0,
    payment_lag_days=0,
    is_weekend_order=0,
    freight_value=12.50,
    rag_context="""
    Distance Guidelines: Deliveries under 100km use standard local carriers.
    Weight Rules: Lightweight packages (under 500g) have 99% on-time rate.
    Weekday Orders: Regular business hours ensure fast processing.
    Payment: Immediate payment reduces risk to near-zero.
    Carrier: Standard shipping sufficient; no upgrades needed.
    """
)


# ===== SCENARIO 3: MODERATE-RISK BORDERLINE CASE =====
SCENARIO_3_MODERATE_RISK = DeliveryScenario(
    predicted_days=6.5,
    promised_days=6.0,
    distance_km=650.0,
    weight_g=2000.0,
    payment_lag_days=3,
    is_weekend_order=0,
    freight_value=65.00,
    rag_context="""
    Distance Guidelines: Regional delivery 500-1000km typically takes 5-7 days.
    Weight Rules: Medium packages (1.5-2.5kg) standard on regional routes.
    Payment Lag: 3-day lag creates moderate risk (15-20% penalty).
    Carrier Options: Regional carriers (SEDEX) recommended for 500-1000km.
    Recovery: Vouchers (15%) effective if delay occurs; retain 80% customers.
    """
)


# ===== TEST RUNNER =====
def run_scenario_test(scenario: DeliveryScenario, scenario_name: str) -> Dict:
    """
    Run full multi-agent analysis on scenario and return results
    
    Args:
        scenario: DeliveryScenario instance
        scenario_name: Name for logging
        
    Returns:
        Dictionary with full analysis results
    """
    print(f"\n{'='*70}")
    print(f"SCENARIO: {scenario_name}")
    print(f"{'='*70}")
    print(f"Distance: {scenario.distance_km}km | Weight: {scenario.weight_g}g")
    print(f"Predicted: {scenario.predicted_days} days | Promised: {scenario.promised_days} days")
    print(f"Payment Lag: {scenario.payment_lag_days} days | Weekend: {'Yes' if scenario.is_weekend_order else 'No'}")
    print()
    
    # Run multi-agent analysis
    result: IntegratedDecision = run_multi_agent_analysis_parallel(scenario)
    
    # Format and display results
    print("\n📊 RISK ASSESSMENT:")
    print(f"  Level: {result.risk_assessment.risk_level}")
    print(f"  Score: {result.risk_assessment.risk_score}/100")
    print(f"  Factors: {', '.join(result.risk_assessment.primary_risk_factors)}")
    print(f"  Analysis: {result.risk_assessment.analysis[:150]}...")
    
    print("\n🚚 CARRIER RECOMMENDATION:")
    print(f"  Recommended: {result.carrier_recommendation.recommended_carrier}")
    print(f"  Current: {result.carrier_recommendation.current_carrier}")
    print(f"  Upgrade: {result.carrier_recommendation.should_upgrade}")
    print(f"  Cost Impact: R${result.carrier_recommendation.cost_impact}")
    print(f"  ROI: {result.carrier_recommendation.roi_analysis[:120]}...")
    
    print("\n💰 RECOVERY STRATEGY:")
    print(f"  Voucher: {result.recovery_plan.voucher_code or 'None'}")
    print(f"  Discount: {result.recovery_plan.discount_percentage}%")
    print(f"  Retention: {result.recovery_plan.retention_probability}%")
    print(f"  Timing: {result.recovery_plan.timing}")
    
    print("\n🎯 FINAL DECISION:")
    print(f"  Executive Summary: {result.executive_summary[:200]}...")
    print(f"  Est. Delivery: {result.estimated_delivery_time} days")
    print(f"  Confidence: {result.confidence_score}%")
    
    return {
        "scenario_name": scenario_name,
        "risk_level": result.risk_assessment.risk_level,
        "risk_score": result.risk_assessment.risk_score,
        "recommended_carrier": result.carrier_recommendation.recommended_carrier,
        "should_upgrade": result.carrier_recommendation.should_upgrade,
        "voucher": result.recovery_plan.voucher_code,
        "discount": result.recovery_plan.discount_percentage,
        "retention": result.recovery_plan.retention_probability,
        "confidence": result.confidence_score
    }


def run_all_scenarios():
    """Run all 3 scenarios and compare results"""
    print("\n" + "="*70)
    print("AGENTICAI LOGISTICS - 3 SCENARIO TEST SUITE")
    print("="*70)
    
    results = []
    
    # Run scenarios
    results.append(run_scenario_test(SCENARIO_1_HIGH_RISK, "HIGH-RISK: Long Distance + Heavy + Weekend"))
    results.append(run_scenario_test(SCENARIO_2_LOW_RISK, "LOW-RISK: Local Delivery + Light + Weekday"))
    results.append(run_scenario_test(SCENARIO_3_MODERATE_RISK, "MODERATE-RISK: Regional + Medium + Lag"))
    
    # Summary comparison
    print("\n" + "="*70)
    print("SUMMARY COMPARISON")
    print("="*70)
    print(f"{'Scenario':<30} {'Risk':<12} {'Carrier':<20} {'Voucher':<12} {'Conf%':<8}")
    print("-"*70)
    for r in results:
        print(f"{r['scenario_name']:<30} {r['risk_level']:<12} {r['recommended_carrier']:<20} {r['voucher'] or 'None':<12} {r['confidence']:<8}")


if __name__ == "__main__":
    run_all_scenarios()
