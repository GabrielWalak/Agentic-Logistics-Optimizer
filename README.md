# Multi-Agent Logistics AI System

Portfolio implementation of a multi-step logistics decision-support workflow. Four specialized LLM roles combine an XGBoost prediction, retrieved logistics knowledge, and typed carrier quotes to produce an evaluated recommendation.

## Live Demo

**[https://gwprojects.switzerlandnorth.cloudapp.azure.com/](https://gwprojects.switzerlandnorth.cloudapp.azure.com/)** — deployed on Azure with HTTPS (Let's Encrypt)


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
- **Carrier quote tool** supplies validated price, availability, and transit data to Agent 2
- **Agent 4** integrates all outputs into a cohesive action plan
- **Response Grader** validates logical correctness (not just JSON format)

---

## Tech Stack

| Layer | Technology |
|-------|-----------|
| **LLM** | Gemini Flash via OpenAI-compatible Gemini API |
| **Framework** | FastAPI + Pydantic V2 |
| **Architecture** | Multi-Agent Orchestration (4 agents) |
| **Knowledge Base** | RAG + ChromaDB Vector Store |
| **Database** | PostgreSQL + SQLModel (async) |
| **Cache** | Redis (response caching) |
| **Infrastructure** | Docker Compose + Azure VM + GitHub Actions |
| **Observability** | LangSmith + Structured JSON Logging |
| **Evaluation** | Behavioral Grading Framework |
| **Security** | API Key Auth + Basic Auth (portfolio) |
| **Prompt Engineering** | Centralized constrained prompts with JSON contracts |

---

## Key Features

- **Real-time AI Analysis** — 4 LLM calls in ~12-15s with parallel execution
- **Behavioral Grading** — validates logic consistency, not just output format (score-level alignment, factor specificity, ROI analysis)
- **Bounded LLM Calls** — client timeout, workflow timeout, and 3 application-level retry attempts
- **Typed Carrier Tool** — Pydantic contracts keep price, availability, and transit data deterministic
- **RAG Integration** — 6 logistics knowledge base documents for context-aware decisions
- **Live Demo** — interactive portfolio page with 3 predefined scenarios (HIGH/MODERATE/LOW risk)
- **Audit Trail** — PostgreSQL JSON storage for successful authenticated analyses

---

## API Endpoints

| Method | Endpoint | Auth | Description |
|--------|----------|------|-------------|
| `GET` | `/` | Basic Auth | Portfolio page with live demo |
| `POST` | `/analyze` | API Key | Full multi-agent analysis |
| `POST` | `/batch-analyze` | API Key | Sequential batch processing with per-item results |
| `POST` | `/demo/analyze` | None | Public demo (predefined scenarios only) |
| `GET` | `/health` | None | Health check |
| `GET` | `/status` | None | Runtime metrics |
| `GET` | `/debug/llm-test` | API Key | Authenticated LLM connectivity probe |
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
      "estimated_cost": 96.1,
      "estimated_transit_days": 4.0,
      "quote_source": "portfolio_rate_card_v1",
      "roi_analysis": "Verified cost impact is R$50.00; full ROI requires validated penalty data."
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

**Known limitations:** The typed carrier tool uses a deterministic portfolio rate card, not a live carrier API. The model does not account for real-time weather, fleet availability, traffic disruptions, or holiday surges. The tool contract can be reused with a live API or MCP implementation.

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
# Edit .env with your Gemini API key from Google AI Studio

# 5. Run the API
uvicorn main:app --reload
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

Production releases are automated by GitHub Actions. Pull requests run tests,
static checks, and a Docker build. A successful push to `main` publishes an
immutable release to the existing Azure VM through SSH, builds the image on
the VM, and performs health verification with automatic rollback. No additional
Azure Container Registry is required.

See [DEPLOYMENT.md](DEPLOYMENT.md) for the one-time Azure and GitHub setup.

---

## Project Structure

```
├── main.py                    # FastAPI server + portfolio page
├── pydantic_agents.py         # Multi-agent orchestration (4 agents)
├── prompt_engineering.py      # Canonical prompts + ResponseGrader
├── carrier_tools.py           # Typed deterministic carrier quote tool
├── logistics_knowledge_base.py # RAG document loader
├── chroma_db_manager.py       # ChromaDB vector store
├── models.py                  # SQLModel tables (audit logs)
├── database.py                # Async PostgreSQL connection
├── docker-compose.yml         # PostgreSQL + Redis + App
├── Dockerfile                 # Production container
├── deploy/                    # Azure VM Compose, bootstrap and rollback scripts
├── .github/workflows/ci.yml  # CI plus Azure production deployment
└── logistics_docs/            # Knowledge base documents
```
