# Multi-Agent Logistics AI System

Enterprise-grade multi-agent AI system for real-time logistics decision-making. Four specialized LLM agents work in parallel to analyze delivery scenarios, optimize carrier selection, design customer recovery strategies, and produce integrated decisions with confidence scoring.

## Live Demo

**[https://gwprojects.tech](https://gwprojects.switzerlandnorth.cloudapp.azure.com/)** — deployed on Azure with HTTPS (Let's Encrypt)


---

## Architecture

```
Delivery Scenario Input (distance, weight, time, payment)
        │
        ▼
┌─ Agent 1: Risk Assessment ──────────── Score: 0-100, risk factors
│       │
│       ├── Agent 2: Carrier Optimization ── ROI analysis (parallel)
│       │
│       └── Agent 3: Recovery Strategy ───── Voucher logic (parallel)
│               │
│               ▼
└─ Agent 4: Decision Orchestrator ────── Executive summary + confidence
                │
                ▼
        Response Grader ──────────────── Behavioral validation (0-100)
                │
                ▼
        Final Decision + Grading Score
```

- **Agent 1** runs first (risk assessment feeds into other agents)
- **Agents 2 & 3** run in parallel via `ThreadPoolExecutor`
- **Agent 4** integrates all outputs into a cohesive action plan
- **Response Grader** validates logical correctness (not just JSON format)

---

## Tech Stack

| Layer | Technology |
|-------|-----------|
| **LLM** | GPT-4o-mini via GitHub Models API |
| **Framework** | FastAPI + Pydantic V2 |
| **Architecture** | Multi-Agent Orchestration (4 agents) |
| **Knowledge Base** | RAG + ChromaDB Vector Store |
| **Database** | PostgreSQL + SQLModel (async) |
| **Cache** | Redis (response caching) |
| **Infrastructure** | Docker + Azure |
| **Observability** | LangSmith + Structured JSON Logging |
| **Evaluation** | Behavioral Grading Framework |
| **Security** | API Key Auth + Basic Auth (portfolio) |
| **Prompt Engineering** | V2 Optimized with constraint validation |

---

## Key Features

- **Real-time AI Analysis** — 4 LLM calls in ~12-15s with parallel execution
- **Behavioral Grading** — validates logic consistency, not just output format (score-level alignment, factor specificity, ROI analysis)
- **Retry Logic** — exponential backoff with 3 attempts per LLM call
- **RAG Integration** — 6 logistics knowledge base documents for context-aware decisions
- **Live Demo** — interactive portfolio page with 3 predefined scenarios (HIGH/MODERATE/LOW risk)
- **Audit Trail** — PostgreSQL JSONB storage for all requests/responses

---

## API Endpoints

| Method | Endpoint | Auth | Description |
|--------|----------|------|-------------|
| `GET` | `/` | Basic Auth | Portfolio page with live demo |
| `POST` | `/analyze` | API Key | Full multi-agent analysis |
| `POST` | `/batch-analyze` | API Key | Batch processing (up to 100) |
| `POST` | `/demo/analyze` | None | Public demo (predefined scenarios only) |
| `GET` | `/health` | None | Health check |
| `GET` | `/status` | None | Runtime metrics |
| `GET` | `/docs` | None | Swagger UI |

---

## Response Example

```json
{
  "request_id": "14a8b690-...",
  "decision": {
    "risk_assessment": {
      "risk_level": "HIGH",
      "risk_score": 75.0,
      "primary_risk_factors": ["Long distance (2800km)", "Heavy weight (4500g)", "Weekend order"],
      "analysis": "The predicted delivery time of 12.5 days significantly exceeds..."
    },
    "carrier_recommendation": {
      "recommended_carrier": "Premium Express",
      "should_upgrade": true,
      "cost_impact": 50.0,
      "roi_analysis": "ROI = (200 - 50) / 50 = 300%. Strong financial justification."
    },
    "recovery_plan": {
      "voucher_code": "DELAY25",
      "discount_percentage": 25.0,
      "retention_probability": 70.0
    },
    "executive_summary": "Risk level HIGH (75/100). Carrier upgrade to Premium Express recommended...",
    "confidence_score": 85.0
  },
  "grading": {
    "overall_score": 97.7,
    "quality_level": "Excellent"
  },
  "processing_time_ms": 12500
}
```

---

## Grading Framework

The system evaluates AI response quality across multiple dimensions:

| Dimension | Points | Validates |
|-----------|--------|-----------|
| Score-Level Alignment | 25 | risk_score matches risk_level range |
| Factor Specificity | 20 | Measurable factors with units (km, kg, days) |
| Logic Consistency | 20 | upgrade=true → cost>0, discount matches voucher |
| ROI Analysis | 25 | Numerical cost-benefit calculation |
| Analysis Depth | 15 | >80 words, minimum 2 sentences |

---

## ML Model Context

The delivery time prediction model is trained on the **Brazilian E-Commerce (Olist) dataset** (~100k orders, 2016-2018).

**Known limitations:** Does not account for real-time weather, carrier fleet availability, traffic disruptions, or holiday surges. In production, the system would integrate live carrier APIs and weather data.

---

## Local Development

```bash
# 1. Clone and setup
git clone https://github.com/GabrielWalak/Agentic-Logistics-Optimizer.git
cd Agentic-Logistics-Optimizer

# 2. Create virtual environment
python -m venv .venv
source .venv/bin/activate  # Linux/Mac
# .venv\Scripts\activate   # Windows

# 3. Install dependencies
pip install -r requirements.txt

# 4. Configure environment
cp .env.example .env
# Edit .env with your GITHUB_TOKEN (needs 'models' permission)

# 5. Run locally (without PostgreSQL)
python _run_local.py
```

---

## Production Deployment (Docker)

```bash
# Configure .env with production values
cp .env.example .env

# Build and run
docker-compose up -d --build

# Check status
docker logs agentic-logistics-api --tail 20
```

**Requirements:** Docker, Docker Compose, GitHub PAT with `models` permission.

---

## Project Structure

```
├── main.py                    # FastAPI server + portfolio page
├── pydantic_agents.py         # Multi-agent orchestration (4 agents)
├── prompt_engineering.py      # Prompts V2 + ResponseGrader
├── logistics_knowledge_base.py # RAG document loader
├── chroma_db_manager.py       # ChromaDB vector store
├── models.py                  # SQLModel tables (audit logs)
├── database.py                # Async PostgreSQL connection
├── docker-compose.yml         # PostgreSQL + Redis + App
├── Dockerfile                 # Production container
└── logistics_docs/            # Knowledge base documents
```
