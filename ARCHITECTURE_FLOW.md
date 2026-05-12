# Complete Architecture Flow: Where API Gets Called

## 🎯 Entry Point → API Call → Result

### START: User runs app.py

```bash
python app.py
```

**app.py line 150+:**
```python
class AgenticAIApp:
    def run_single_scenario(self):
        # User picks scenario (e.g., "1" for HIGH-RISK)
        scenario_name, scenario = self.scenarios[choice]  # Gets SCENARIO_1_HIGH_RISK
        result = run_scenario_test(scenario, scenario_name)
        return result
```

---

## 🔄 Flow: app.py → scenarios_examples.py → pydantic_agents.py

### STEP 1: Choose Scenario in app.py (Menu Option 1)

```python
# app.py line 62-70
def run_single_scenario(self):
    choice = input("Enter scenario number (1-3): ")  # User enters "1"
    scenario_name, scenario = self.scenarios[choice]  # = SCENARIO_1_HIGH_RISK
```

**SCENARIO_1_HIGH_RISK from scenarios_examples.py (line 10-28):**
```python
SCENARIO_1_HIGH_RISK = DeliveryScenario(
    predicted_days=12.5,
    promised_days=7.0,
    distance_km=2800,        # ← LONG DISTANCE
    weight_g=4500,           # ← HEAVY WEIGHT
    payment_lag_days=5,
    is_weekend_order=1,      # ← WEEKEND
    freight_value=150.00,
    rag_context="Distance Guidelines: Deliveries over 2000km require premium carriers..."
)
```

---

### STEP 2: Call Multi-Agent Orchestrator

**app.py line 70:**
```python
result = run_scenario_test(scenario, scenario_name)
```

**scenarios_examples.py line 100+ (run_scenario_test):**
```python
def run_scenario_test(scenario: DeliveryScenario, scenario_name: str) -> Dict:
    print(f"\n{'='*80}")
    print(f" {scenario_name}")
    print(f"{'='*80}\n")
    
    # ← HERE: CALL THE MULTI-AGENT SYSTEM
    result: IntegratedDecision = run_multi_agent_analysis_parallel(scenario)
```

---

### STEP 3: MAIN FLOW - pydantic_agents.py (Line 540+)

**pydantic_agents.py line 540-556:**
```python
@traceable(name="multi_agent_analysis")
def run_multi_agent_analysis_parallel(scenario: DeliveryScenario) -> IntegratedDecision:
    """
    THIS IS WHERE THE MAGIC HAPPENS
    scenario = DeliveryScenario with distance_km=2800, weight_g=4500, etc
    """
    start_time = time.time()
    print("🤖 Starting multi-agent analysis (parallel)...")
    
    # AGENT 1: RISK ASSESSMENT (Line 558-559)
    print("  ├─ Agent 1: Risk Assessment...")
    risk_dict = run_risk_assessment(scenario)  # ← FIRST API CALL
    
    # AGENTS 2-3: PARALLEL (Line 562-567)
    print("  ├─ Agents 2-3: Parallel execution...")
    with concurrent.futures.ThreadPoolExecutor(max_workers=2) as executor:
        carrier_future = executor.submit(run_carrier_optimization, scenario, risk_dict)  # ← PARALLEL API CALL
        recovery_future = executor.submit(run_recovery_strategy, scenario, risk_dict)    # ← PARALLEL API CALL
        
        carrier_dict = carrier_future.result()
        recovery_dict = recovery_future.result()
    
    # AGENT 4: ORCHESTRATOR (Line 571-572)
    print("  └─ Agent 4: Decision Integration...")
    orchestrator_dict = run_orchestrator(scenario, risk_dict, carrier_dict, recovery_dict)  # ← FINAL API CALL
```

---

## 🌐 WHERE THE ACTUAL API CALLS HAPPEN

### Agent 1: Risk Assessment (pydantic_agents.py line 295-330)

```python
@traceable(name="risk_assessment_agent")
def run_risk_assessment(scenario: DeliveryScenario) -> Dict:
    """Agent 1: Evaluates delivery risk"""
    
    # Build user prompt from scenario
    delay_days = scenario.predicted_days - scenario.promised_days  # 12.5 - 7 = 5.5 days
    
    user_prompt = f"""Analyze this delivery scenario:
    
Distance: {scenario.distance_km}km (2800)
Weight: {scenario.weight_g}g (4500)
Delay: {delay_days:.1f} days (5.5)
Payment Lag: {scenario.payment_lag_days} days
Weekend Order: {scenario.is_weekend_order == 1}

Knowledge Base:
{scenario.rag_context}

Provide risk assessment in JSON..."""
    
    # ← HERE: CALLS GITHUB MODELS API via call_ollama()
    response = call_ollama(RISK_AGENT_PROMPT_V2, user_prompt)
    #         ^^^^^^^^^^^ 
    #         THIS CALLS: pydantic_agents.py line 107-154
    
    result = parse_json_response(response)  # Parse JSON from LLM
    return result
```

---

### The Actual API Call: call_ollama() (pydantic_agents.py line 107-154)

```python
def call_ollama(
    system_prompt: str,      # = RISK_AGENT_PROMPT_V2
    user_prompt: str,        # = "Analyze this delivery scenario: Distance 2800km..."
    model: Optional[str] = None,
    stream: bool = False
) -> str:
    """LLM API call via GitHub Models using OpenAI SDK"""
    
    try:
        # Step 1: Check cache (Redis optional)
        cache_key = rag_cache.make_key(system_prompt + user_prompt)
        cached = rag_cache.get(cache_key)
        if cached:
            print("[cached] ", end="", flush=True)
            return cached
        
        # Step 2: Get LLM client
        model_name = model or os.getenv("GITHUB_MODEL", "gpt-4o-mini")
        client = _get_github_client()  # ← Creates OpenAI client for GitHub Models
        
        # ← ← ← ACTUAL API CALL ← ← ←
        response = client.chat.completions.create(
            model=model_name,  # "gpt-4o-mini"
            messages=[
                {"role": "system", "content": system_prompt},      # RISK_AGENT_PROMPT_V2
                {"role": "user", "content": user_prompt},          # Scenario data
            ],
            temperature=0.3,    # Deterministic
            max_tokens=1024,
            top_p=0.9,
        )
        
        # Step 3: Extract response
        result = response.choices[0].message.content or ""
        
        # Step 4: Cache for next time
        rag_cache.set(cache_key, result)
        
        return result
        
    except Exception as e:
        return f'{{"error": "GitHub Models error: {str(e)}"}}'
```

---

## 📊 CONCRETE EXAMPLE: HIGH-RISK SCENARIO

### What happens step-by-step:

```
User runs: python app.py → Menu Option 1
     ↓
Scenario chosen: "1" (HIGH-RISK)
     ↓
SCENARIO_1_HIGH_RISK = {
    predicted_days: 12.5,
    promised_days: 7.0,
    distance_km: 2800,
    weight_g: 4500,
    ...
}
     ↓
run_scenario_test(SCENARIO_1_HIGH_RISK, "HIGH-RISK: Long Distance...")
     ↓
run_multi_agent_analysis_parallel(SCENARIO_1_HIGH_RISK)
     ↓
     
AGENT 1 API CALL:
─────────────────
call_ollama(
    system_prompt = RISK_AGENT_PROMPT_V2,  # "You are a Risk Assessment Specialist..."
    user_prompt = "Distance: 2800km\nWeight: 4500g\nDelay: 5.5 days\n..."
)
→ GitHub Models API (gpt-4o-mini)
← Response: {
    "risk_level": "HIGH",
    "risk_score": 78,
    "primary_risk_factors": ["Long distance (2800km)", "Heavy weight (4500g)", "Weekend order"],
    "mitigation_priority": "URGENT",
    "analysis": "2800km exceeds 2000km threshold requiring premium carrier..."
  }

     ↓
AGENTS 2-3 RUN IN PARALLEL:
──────────────────────────

Carrier Agent calls:
call_ollama(
    system_prompt = CARRIER_AGENT_PROMPT_V2,
    user_prompt = "Risk: HIGH (78)\nDistance: 2800km\nRecommend carrier..."
)
→ GitHub Models API
← Response: {
    "recommended_carrier": "Premium Express",
    "should_upgrade": true,
    "cost_impact": 50,
    ...
  }

Recovery Agent calls:
call_ollama(
    system_prompt = RECOVERY_AGENT_PROMPT_V2,
    user_prompt = "Risk: HIGH\nDelay: 5.5 days\nDesign recovery..."
)
→ GitHub Models API
← Response: {
    "voucher_code": "DELAY25",
    "discount_percentage": 25,
    ...
  }

     ↓
AGENT 4 ORCHESTRATOR:
────────────────────
call_ollama(
    system_prompt = ORCHESTRATOR_PROMPT,
    user_prompt = "Risk: HIGH (78)\nCarrier: Upgrade to Premium (+$50)\nVoucher: DELAY25 (25%)..."
)
→ GitHub Models API
← Response: {
    "executive_summary": "Upgrade to Premium Express + send DELAY25 voucher on Day 1...",
    "confidence_score": 88,
    ...
  }

     ↓
GRADING (prompt_engineering.py):
────────────────────────────────
ResponseGrader validates each response:
- Risk: 78 ✓ HIGH range? YES (61-80) → 25/25 points
- Carrier: cost_impact > 0 when upgrade=true? YES → 20/20 points
- Recovery: DELAY25 matches 25% discount? YES → 15/15 points

Final Score: 85/100 ✅
```

---

## 🎯 KEY INSIGHTS

| Part | File | Line | What It Does |
|------|------|------|-------------|
| **User Interface** | app.py | 62-70 | Menu → Scenario selection |
| **Scenario Definition** | scenarios_examples.py | 10-28 | Data: 2800km, 4500g, etc |
| **Test Runner** | scenarios_examples.py | 100+ | Calls multi-agent analysis |
| **Main Orchestrator** | pydantic_agents.py | 540-580 | Coordinates 4 agents in parallel |
| **Agent 1: Risk** | pydantic_agents.py | 295-330 | Calls API with risk prompt |
| **Agent 2: Carrier** | pydantic_agents.py | 337-380 | Calls API with carrier prompt |
| **Agent 3: Recovery** | pydantic_agents.py | 385-430 | Calls API with recovery prompt |
| **Agent 4: Orchestrator** | pydantic_agents.py | 433-500 | Calls API to integrate all 3 |
| **Actual API Call** | pydantic_agents.py | 107-154 | `call_ollama()` → GitHub Models |
| **Response Parsing** | pydantic_agents.py | 158-175 | JSON extraction from LLM |
| **Grading** | prompt_engineering.py | 95-250 | Validates logic (score 85/100) |

---

## 🔗 EXECUTION CHAIN

```
app.py (Menu)
    ↓
scenarios_examples.py (run_scenario_test)
    ↓
pydantic_agents.py (run_multi_agent_analysis_parallel)
    ├─ run_risk_assessment()
    │   └─ call_ollama(RISK_AGENT_PROMPT_V2, user_prompt)
    │       └─ client.chat.completions.create()  ← GITHUB MODELS API
    │
    ├─ ThreadPoolExecutor (parallel)
    │   ├─ run_carrier_optimization()
    │   │   └─ call_ollama(CARRIER_AGENT_PROMPT_V2, ...)  ← API
    │   └─ run_recovery_strategy()
    │       └─ call_ollama(RECOVERY_AGENT_PROMPT_V2, ...)  ← API
    │
    └─ run_orchestrator()
        └─ call_ollama(ORCHESTRATOR_PROMPT, ...)  ← API
    
    ↓
prompt_engineering.py (ResponseGrader)
    ├─ grade_risk_assessment()
    ├─ grade_carrier_optimization()
    └─ grade_recovery_strategy()
    
    ↓
Result: IntegratedDecision (all graded)
```

---

## 💡 WHAT YOU NEED TO UNDERSTAND

1. **scenario_examples.py defines scenarios** - distance, weight, delay, payment lag
2. **pydantic_agents.py has 4 agent functions** - risk, carrier, recovery, orchestrator
3. **Each agent calls `call_ollama()`** - that's the GitHub Models API call
4. **`call_ollama()` does:**
   - Create OpenAI client pointed to GitHub Models endpoint
   - Send system prompt (teaches LLM the rules)
   - Send user prompt (scenario data)
   - Get JSON response back
   - Cache result for efficiency
5. **4 API calls run (3 in parallel, 1 sequential)** - total 10-20 seconds
6. **ResponseGrader validates each response** - checks logic, not just JSON format
7. **Final score 0-100** - reflects quality of AI reasoning

---

**Want to trace through the code yourself?** Start here:
- User selects scenario: [app.py:62](app.py#L62)
- Scenario sent to orchestrator: [scenarios_examples.py:100](scenarios_examples.py#L100)
- Orchestrator runs 4 agents: [pydantic_agents.py:540](pydantic_agents.py#L540)
- Each agent calls API: [pydantic_agents.py:107](pydantic_agents.py#L107)
