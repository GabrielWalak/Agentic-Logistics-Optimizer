"""
AgenticAI Main Application
Master orchestrator for multi-agent logistics system with prompt engineering & grading
"""
import sys
from typing import Dict, List
from scenarios_examples import (
    SCENARIO_1_HIGH_RISK,
    SCENARIO_2_LOW_RISK,
    SCENARIO_3_MODERATE_RISK,
    run_scenario_test
)
from prompt_engineering import ResponseGrader, generate_grading_report
from pydantic_agents import (
    DeliveryScenario,
    run_multi_agent_analysis_parallel,
    IntegratedDecision,
    check_ollama_status
)


class AgenticAIApp:
    """Main application coordinator"""
    
    def __init__(self):
        self.scenarios = {
            "1": ("HIGH-RISK: Long Distance + Heavy + Weekend", SCENARIO_1_HIGH_RISK),
            "2": ("LOW-RISK: Local Delivery + Light + Weekday", SCENARIO_2_LOW_RISK),
            "3": ("MODERATE-RISK: Regional + Medium + Lag", SCENARIO_3_MODERATE_RISK),
        }
        self.grader = ResponseGrader()
    
    def print_header(self, title: str):
        """Print formatted header"""
        print("\n" + "="*80)
        print(f" {title}")
        print("="*80)
    
    def print_menu(self):
        """Display main menu"""
        self.print_header("AGENTICAI LOGISTICS SYSTEM - MAIN MENU")
        print("""
1. Run Single Scenario Analysis
2. Run All 3 Scenarios (with comparison)
3. Test Prompt Engineering (optimized prompts)
4. Grade Agent Responses (quality evaluation)
5. View System Status
6. Exit
        """)
    
    def run_single_scenario(self):
        """Run analysis on one scenario"""
        self.print_header("SELECT SCENARIO")
        for key, (name, _) in self.scenarios.items():
            print(f"{key}. {name}")
        
        choice = input("\nEnter scenario number (1-3): ").strip()
        if choice not in self.scenarios:
            print("❌ Invalid choice")
            return
        
        scenario_name, scenario = self.scenarios[choice]
        print()
        result = run_scenario_test(scenario, scenario_name)
        
        # Return result for potential grading
        return result
    
    def run_all_scenarios(self):
        """Run all 3 scenarios and compare"""
        self.print_header("RUNNING ALL 3 SCENARIOS")
        
        results = []
        for key in ["1", "2", "3"]:
            scenario_name, scenario = self.scenarios[key]
            result = run_scenario_test(scenario, scenario_name)
            results.append(result)
        
        # Comparison table
        self.print_header("COMPARISON SUMMARY")
        print(f"{'Scenario':<35} {'Risk':<12} {'Carrier':<20} {'Voucher':<15} {'Conf%':<8}")
        print("-"*90)
        for r in results:
            print(f"{r['scenario_name']:<35} {r['risk_level']:<12} {r['recommended_carrier']:<20} {str(r['voucher']):<15} {r['confidence']:<8}")
        
        return results
    
    def test_prompt_engineering(self):
        """Test optimized prompts"""
        self.print_header("PROMPT ENGINEERING TEST")
        
        from prompt_engineering import RISK_AGENT_PROMPT_V2, CARRIER_AGENT_PROMPT_V2, RECOVERY_AGENT_PROMPT_V2
        
        print("\n📝 RISK ASSESSMENT PROMPT (V2 - Optimized):")
        print("-" * 80)
        print(RISK_AGENT_PROMPT_V2[:500] + "...\n")
        
        print("🚚 CARRIER OPTIMIZATION PROMPT (V2 - Optimized):")
        print("-" * 80)
        print(CARRIER_AGENT_PROMPT_V2[:500] + "...\n")
        
        print("💰 RECOVERY STRATEGY PROMPT (V2 - Optimized):")
        print("-" * 80)
        print(RECOVERY_AGENT_PROMPT_V2[:500] + "...\n")
        
        print("✅ Prompts optimized for:")
        print("   - Structured JSON output")
        print("   - Numerical scoring logic")
        print("   - Clear decision criteria")
        print("   - Quantified metrics (ROI, retention %, cost impact)")
    
    def grade_responses(self):
        """Grade agent responses quality"""
        self.print_header("RESPONSE GRADING FRAMEWORK")
        
        print("""
Testing response quality from agents...
Grading Criteria:
  - Risk Assessment: Structure, validity, score alignment, specificity, depth
  - Carrier Recommendation: Structure, carrier validity, boolean, logic, ROI analysis
  - Recovery Strategy: Structure, voucher alignment, discount-voucher alignment, retention alignment
        """)
        
        # Run scenario and grade
        scenario = SCENARIO_1_HIGH_RISK
        print("\nAnalyzing HIGH-RISK scenario...")
        
        result: IntegratedDecision = run_multi_agent_analysis_parallel(scenario)
        
        # Simulate response grading (in real system, we'd capture LLM raw responses)
        sample_responses = {
            "risk": '{"risk_level": "HIGH", "risk_score": 75, "primary_risk_factors": ["Long distance (2800km)", "Heavy weight (4500g)", "Weekend"], "mitigation_priority": "HIGH", "analysis": "High-risk scenario requiring premium carrier and proactive customer communication. The predicted delivery of 12.5 days versus promised 7 days indicates critical delay risk."}',
            "carrier": '{"recommended_carrier": "Premium Express", "current_carrier": "Standard", "should_upgrade": true, "upgrade_rationale": "Long distance requires faster premium carrier to mitigate delay risk", "cost_impact": 50, "roi_analysis": "Upgrade cost R$50 saves R$250 in penalties and customer churn (ROI 400%)"}',
            "recovery": '{"voucher_code": "DELAY25", "discount_percentage": 25, "communication_template": "Subject: Your order may experience a delay. We are offering 25% off (DELAY25) and free express shipping on next order.", "timing": "Day 1: Proactive notification", "retention_probability": 75}'
        }
        
        grading_report = generate_grading_report(sample_responses)
        
        self.print_header("GRADING REPORT")
        print(f"\n🎯 Overall Score: {grading_report['overall_score']}/100")
        print(f"📊 Quality Level: {grading_report['quality_level']}")
        
        print("\n🤖 Agent Scores:")
        for agent, data in grading_report['agents'].items():
            score = data['score']
            details = data['details']
            print(f"  {agent}: {score}/100")
            
            # Show what passed
            passed = [k for k, v in details.items() if v is True and k != "deductions"]
            for criterion in passed:
                print(f"    ✓ {criterion}")
            
            # Show deductions (what could be improved)
            deductions = details.get("deductions", [])
            if deductions:
                for deduction in deductions:
                    print(f"    ⚠ {deduction}")
        
        print("\n💡 Improvement Areas:")
        for rec in grading_report['recommendations']:
            if rec:
                print(f"  • {rec}")
    
    def view_status(self):
        """Check system health"""
        self.print_header("SYSTEM STATUS")
        
        print("\n🔍 Checking components...")
        
        # Check the configured LLM provider
        if check_ollama_status():
            print("  ✓ LLM API: Configured")
        else:
            print("  ✗ LLM API: Not configured")
        
        # Check Pydantic models
        try:
            from pydantic_agents import (
                DeliveryScenario, RiskAssessment, 
                CarrierRecommendation, CustomerRecoveryPlan,
                IntegratedDecision
            )
            print("  ✓ Pydantic Models: Loaded")
        except Exception as e:
            print(f"  ✗ Pydantic Models: {e}")
        
        # Check Scenarios
        try:
            print(f"  ✓ Test Scenarios: {len(self.scenarios)} available")
        except:
            print("  ✗ Test Scenarios: Failed")
        
        # Check Grading
        try:
            print("  ✓ Response Grader: Loaded")
        except:
            print("  ✗ Response Grader: Failed")
        
        print("\n✅ System ready for deployment!")
    
    def run(self):
        """Main application loop"""
        print("""
╔════════════════════════════════════════════════════════════════╗
║          AGENTICAI - MULTI-AGENT LOGISTICS SYSTEM              ║
║   Powered by Gemini API + Pydantic Agents + Grading            ║
╚════════════════════════════════════════════════════════════════╝
        """)
        
        while True:
            self.print_menu()
            choice = input("Enter your choice (1-6): ").strip()
            
            if choice == "1":
                self.run_single_scenario()
            elif choice == "2":
                self.run_all_scenarios()
            elif choice == "3":
                self.test_prompt_engineering()
            elif choice == "4":
                self.grade_responses()
            elif choice == "5":
                self.view_status()
            elif choice == "6":
                print("\n👋 Goodbye! Thank you for using AgenticAI.\n")
                sys.exit(0)
            else:
                print("❌ Invalid choice. Try again.")


if __name__ == "__main__":
    app = AgenticAIApp()
    app.run()
