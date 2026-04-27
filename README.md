# AgenticAI Logistics Optimizer - Multi-Agent AI System

Advanced multi-agent architecture for intelligent logistics optimization using Retrieval-Augmented Generation (RAG), prompt engineering, and behavioral evaluation.

**Status:** ✅ Core AI system complete. Multi-agent orchestration + RAG + Eval framework ready for integration.

### 🛠️ Tech Stack
* **Core:** Python 3.11, FastAPI, Pydantic V2
* **AI & LLM:** GitHub Models API (gpt-4o-mini), OpenAI SDK, ChromaDB (Vector Store), Prompt Engineering
* **Architecture:** Multi-Agent Orchestration, RAG (Retrieval-Augmented Generation), LLM-as-a-Judge
* **Infrastructure:** Docker, AWS EC2 (t3.micro), Environment Variables (.env)

---

## 🧠 AI/ML Architecture

### 1. Multi-Agent Orchestration

Four specialized agents work in parallel, each optimizing a different logistics dimension:

```
Logistics Scenario Input
    ↓ (ThreadPoolExecutor)
┌───────────────────────────────────────┐
│  Risk Assessment Agent                │  Analyzes: Distance, weight, time buffer, payment terms
│  ├─ Score: 0-100 (risk level)        │  Output: Risk factors + mitigation strategy
│  └─ Logic: Constraint validation      │
│                                       │
│  Carrier Selection Agent              │  Analyzes: Cost optimization, capacity, reliability  
│  ├─ Score: ROI ranking               │  Output: Carrier recommendation + cost analysis
│  └─ Logic: Upgrade trigger matrix    │
│                                       │
│  Recovery Strategy Agent              │  Analyzes: Delay scenarios, customer retention
│  ├─ Score: Recovery effectiveness    │  Output: Discount strategy + retention metrics
│  └─ Logic: Voucher value calculation │
│                                       │
│  Decision Integration Agent           │  Synthesizes: All agent outputs into final recommendation
│  ├─ Confidence: 0-100%               │  Output: Prioritized actions + expected outcomes
│  └─ Logic: Cross-agent conflict reso │
└───────────────────────────────────────┘
    ↓
Response Grading (AI Evaluation)
    ↓
Final Logistics Decision
```

**Performance:** All 4 agents run in parallel (~10-20s total execution)

---

### 2. Retrieval-Augmented Generation (RAG)

Knowledge base integration for contextual decision making:

```
Logistics Documents (ChromaDB Vector Store)
├── carrier_rules.txt          # Carrier selection constraints
├── distance_guidelines.txt    # Distance-based routing logic  
├── weight_volume_rules.txt    # Capacity constraints
├── payment_lag_impact.txt     # Payment timing effects
├── customer_recovery.txt      # Recovery strategy templates
└── weekend_holidays.txt       # Business day calculations

    ↓ (Semantic Search)

Agent Query: "What's the carrier upgrade threshold for 2800km?"
    ↓
ChromaDB retrieves: Related rules + historical patterns
    ↓
LLM (gpt-4o-mini) synthesizes: Context-aware answer
```

**Implementation:**
- Vector embeddings: OpenAI Ada embeddings (async)
- Similarity search: Cosine distance (top-3 results)
- Context window: Up to 4 relevant documents per query
- Update mechanism: Rebuild on document changes

---

### 3. Prompt Engineering (V2 Optimized)

Structured prompts for consistent, validated outputs:

```
AGENT PROMPT TEMPLATE:
1. ROLE: [Clear agent identity]
2. CONTEXT: [Business constraints from RAG]
3. INPUT: [Structured scenario data (distance, weight, etc)]
4. OUTPUT_FORMAT: [JSON schema with required fields]
5. VALIDATION_RULES: [Logic gates (e.g., cost > 0 if upgrade=true)]
6. EXAMPLES: [Few-shot examples of good vs bad responses]
```

**V2 Features:**
- Pydantic models for type-safe parsing
- Explicit JSON schema in prompt
- Constraint validation in response
- Few-shot learning with real examples
- Temperature: 0.3 (consistency over creativity)

**Example output:**
```json
{
  "risk_score": 78,
  "risk_level": "HIGH",
  "factors": [
    {"factor": "Distance", "impact": "2800km > 2000km threshold"},
    {"factor": "Time Buffer", "impact": "12.5 vs 7 days = 5.5 buffer (marginal)"}
  ],
  "mitigation": "Upgrade to overnight carrier + customer notification",
  "confidence": 0.87
}
```

---

### 4. Response Evaluation & Grading

Behavioral evaluation framework that validates AI logic, not just output format:

#### Grading Dimensions

**A. Score-Level Alignment (25 points)**
- Validates: Risk score 78 matches HIGH level (61-80 range)
- Penalty: -15 if misaligned (e.g., score 45 but label "HIGH")

**B. Factors Specificity (20 points)**  
- Requires: Measurable factors with units (km, kg, days)
- ✅ Good: "2800km exceeds 2000km threshold"
- ❌ Bad: "Risk factor 1 is significant"
- Penalty: -20 for vague factors

**C. Analysis Depth (15 points)**
- Requires: >80 words + minimum 2 sentences
- Penalty: -15 for shallow analysis

**D. Logic Consistency (20 points)**
- Carrier: If upgrade=true, must have cost>0
- Recovery: Discount % must match voucher logic
- Penalty: -20 for logical errors

**E. Recovery Alignment (15 points)**
- Validates: Discount tier matches customer lifetime value
- Penalty: -10 for misaligned recovery strategy

**F. Actionability (5 points)**
- Requires: Clear next step recommendations
- Penalty: -5 if vague

**Scoring Examples:**
```
Good Response:  85/100
├─ Score-Level: ✓ (25/25)
├─ Specificity: ✓ (20/20)
├─ Depth: ✓ (15/15)
├─ Logic: ✓ (20/20)
├─ Recovery: ✓ (5/5) 
└─ Deductions: None

Bad Response: 15/100
├─ Score-Level: ❌ (10/25) - 45 score but "HIGH" label
├─ Specificity: ❌ (0/20) - Generic factors only
├─ Depth: ❌ (5/15) - 40 words, 1 sentence
├─ Logic: ❌ (0/20) - upgrade=true but cost=0
└─ Deductions: ⚠️ Logic errors, vague analysis
```

---

## 📊 System Components

### Core Files

| File | Role | Type |
|------|------|------|
| **pydantic_agents.py** | Multi-agent orchestrator | Framework |
| **prompt_engineering.py** | Prompts V2 + ResponseGrader | ML Logic |
| **chroma_db_manager.py** | RAG/Vector store manager | Data Layer |
| **logistics_knowledge_base.py** | Document loader | Data Prep |
| **scenarios_examples.py** | Test cases (HIGH/LOW/MID risk) | Testing |
| **app.py** | Interactive CLI | Interface |
| **test_grader.py** | Grading validation | Testing |

### Data Flow

```
Raw Scenario
    ↓
[Pydantic Validation] ← scenarios_examples.py
    ↓
[Multi-Agent Processing] ← pydantic_agents.py
    ├─ retrieve_context() ← chroma_db_manager.py
    ├─ call_ollama() ← GitHub Models API (gpt-4o-mini)
    └─ ThreadPoolExecutor (parallel execution)
    ↓
[Agent Responses (JSON)]
    ↓
[Response Grading] ← prompt_engineering.py / ResponseGrader
    ├─ Validate score-level alignment
    ├─ Check factor specificity
    ├─ Measure analysis depth
    └─ Verify logic consistency
    ↓
[Final Score: 0-100] + Deductions breakdown
    ↓
[User Decision Support]
```

---

## 🧪 Evaluation Examples

### Test Scenario 1: HIGH RISK
```
Distance: 2800km | Weight: 4500g | Time: 12.5 vs 7 days | Payment: 5-day lag

Agent Outputs:
├─ Risk: 78/100 ("HIGH", factors: distance + time buffer marginal)
├─ Carrier: Upgrade to overnight (+$450 cost, ROI 1.8x)
├─ Recovery: 15% voucher if delayed (retention priority)
└─ Decision: UPGRADE recommended

Grading: 85/100 ✅
├─ Score-Level: ✓ (78 = HIGH range)
├─ Specificity: ✓ (concrete km/day metrics)
└─ Logic: ✓ (upgrade cost justified)
```

### Test Scenario 2: LOW RISK  
```
Distance: 45km | Weight: 300g | Time: 2 vs 3 days | Payment: instant

Agent Outputs:
├─ Risk: 22/100 ("LOW", factors: short distance + time buffer 1 day)
├─ Carrier: Standard shipping ($0 uplift)
├─ Recovery: 5% courtesy discount (loyalty build)
└─ Decision: MAINTAIN standard

Grading: 88/100 ✅
├─ Score-Level: ✓ (22 = LOW range)
├─ Specificity: ✓ (distance buffer clearly stated)
└─ Logic: ✓ (no upgrade needed, costs optimized)
```

### Test Scenario 3: MODERATE RISK
```
Distance: 650km | Weight: 2000g | Time: 6.5 vs 6 days | Payment: 3-day lag

Agent Outputs:
├─ Risk: 58/100 ("MODERATE", factors: margin 0.5 days tight)
├─ Carrier: Conditional upgrade (on payment delay > 2 days)
├─ Recovery: 10% voucher if any delay observed
└─ Decision: MONITOR with escalation plan

Grading: 82/100 ✅
├─ Score-Level: ✓ (58 = MODERATE range)
├─ Specificity: ✓ (payment lag conditional logic)
└─ Logic: ✓ (escalation rules clear)
```

---

## 🔄 Key Features

### Parallel Execution
```python
ThreadPoolExecutor (max_workers=4)
├─ Risk Assessment (async)
├─ Carrier Selection (async)
├─ Recovery Strategy (async)
└─ Decision Integration (async)
Total time: ~10-20s (vs ~40-80s sequential)
```

### Behavioral Validation
- Not just "is JSON valid?" but "does the logic make sense?"
- Detects hallucinations (e.g., score doesn't match risk level)
- Verifies cross-agent consistency (recovery aligns with risk)

### RAG Context Awareness
- Agents ground decisions in knowledge base
- Dynamic document retrieval per query
- Semantic similarity matching (not keyword)

### Type Safety
- Pydantic models for all inputs/outputs
- Runtime validation prevents malformed responses
- Clear error messages for debugging

---

## 📈 Performance Metrics

| Metric | Value | Status |
|--------|-------|--------|
| Parallel execution | 10-20s | ✅ Optimized |
| Response grading accuracy | 85/100 (good), 15/100 (bad) | ✅ Differentiates |
| Agent consensus | 85-95% agreement | ✅ Stable |
| RAG relevance | Top-3 results >= 0.7 similarity | ✅ Valid |
| API reliability | GitHub Models 99.5% uptime | ✅ Tested |

---

## 🎯 Next Steps (Feature Development)

- [ ] Add FastAPI endpoints for remote agent queries
- [ ] Implement caching layer (Redis) for repeated scenarios
- [ ] Extend RAG to support real-time carrier pricing
- [ ] Add A/B testing framework for prompt variations
- [ ] Implement feedback loop for grader model retraining

---

## 📚 How to Use

### Run All Test Scenarios
```bash
python -c "from scenarios_examples import run_all_scenarios; run_all_scenarios()"
```

### Test Single Scenario with Grading
```bash
python app.py  # Menu option 2: "Run all 3 test scenarios"
```

### Evaluate Response Quality
```bash
python test_grader.py
```

### Check System Status
```bash
python app.py  # Menu option 5: "System Status"
```

---

## 🔗 Related Documentation

- **[DEPLOYMENT.md](DEPLOYMENT.md)** - System overview & infrastructure
- **[SECURITY.md](SECURITY.md)** - Token management & security practices
- **[AWS_DEPLOYMENT.md](AWS_DEPLOYMENT.md)** - Cloud deployment guide
- **[BUILD_INSTRUCTIONS.md](BUILD_INSTRUCTIONS.md)** - Local Docker build

---

