# Multi-Agent Logistics AI System

A portfolio implementation of a logistics decision-support workflow. The API combines an XGBoost delivery-time prediction, ChromaDB retrieval, four LLM roles, a typed carrier quote tool, deterministic business rules, and behavioral grading.

The project deliberately uses explicit Python orchestration instead of hiding the workflow behind an agent framework. This keeps control flow, concurrency, failure handling, and data ownership easy to inspect during a technical review.

## Live Demo

**[https://gwprojects.switzerlandnorth.cloudapp.azure.com/](https://gwprojects.switzerlandnorth.cloudapp.azure.com/)** - hosted on an Azure VM with HTTPS.

The portfolio page is protected with Basic Auth. Its three predefined scenarios call the real ML, RAG, and multi-agent workflow. The displayed processing time, specialist scores, combined quality score, and decision details come from the latest API response rather than hard-coded examples.

---

## End-to-End Request Flow

```text
POST /analyze, /batch-analyze, or /demo/analyze
        |
        v
Pydantic request validation
        |
        +-- predicted_days missing? --> XGBoost prediction
        |
        +-- default RAG context? ----> ChromaDB similarity search
        |
        v
Agent 1: Risk Assessment
        |
        +-- application recalculates the authoritative score, level,
        |   priority, and scored factors from deterministic rules
        |
        +-------------------------------+
        |                               |
        v                               v
Agent 2: Carrier Optimization     Agent 3: Recovery Strategy
        |                               |
Typed carrier quote tool               |
        +---------------+---------------+
                        |
                        v
             Agent 4: Decision Orchestrator
                        |
                        v
             Pydantic IntegratedDecision
                        |
                        v
        Deterministic ResponseGrader
        Risk score + Carrier score + Recovery score
                        |
                        v
             arithmetic mean (0-100)
                        |
                        +-- PostgreSQL audit log for /analyze
                        |
                        v
                    API response
```

### Agent orchestration

- **Agent 1 - Risk Assessment:** explains the shipment risk, while Python owns the final numerical score and risk level.
- **Agent 2 - Carrier Optimization:** consumes risk output and deterministic carrier quote data.
- **Agent 3 - Recovery Strategy:** selects a voucher and customer communication strategy.
- **Agent 4 - Decision Orchestrator:** integrates the three specialist outputs into a final recommendation and bounded confidence estimate.
- **ResponseGrader:** scores risk, carrier, and recovery outputs independently; `overall_score` is their arithmetic mean.

Agent 1 runs first because its normalized result is an input to the other specialists. Agents 2 and 3 then run concurrently in a `ThreadPoolExecutor(max_workers=2)`. Agent 4 runs after both futures complete.

---

## Deterministic Risk Grounding

The LLM may explain the risk, but it cannot invent the authoritative score. Python recalculates it from auditable rules:

| Condition | Points |
|-----------|-------:|
| Distance over 1,500 km | +20 |
| Weight over 3,000 g | +15 |
| Predicted delay over 3 days | +25 |
| Payment lag over 5 days | +10 |
| Weekend order | +5 |

Risk levels are mapped consistently: `MINIMAL` 0-20, `LOW` 21-40, `MODERATE` 41-60, `HIGH` 61-80, and `CRITICAL` 81-100.

If an LLM narrative contradicts the ML prediction, states a different score or level, or introduces unsupported percentages/ranges, the application replaces it with a grounded deterministic explanation.

---

## State and Storage

| Scope | Implementation | Purpose |
|-------|----------------|---------|
| Request state | Pydantic V2 models | Typed inputs and agent outputs |
| Process state | `AppState` | In-memory request counters, success/error counts, uptime, average latency |
| Readiness state | `app.state.db_ready` | Refreshed by the live PostgreSQL health probe |
| Durable audit state | PostgreSQL + SQLModel | Successful authenticated `/analyze` inputs, decisions, model name, and latency |
| LLM cache | Redis, 1-hour TTL | Reuses structured LLM responses using prompt/model/schema-aware keys |
| Portfolio response cache | Redis, 10-minute TTL | Reuses a completed response for repeated predefined demo scenarios |
| Vector state | Persistent ChromaDB | Stores embeddings for six logistics knowledge documents |

Redis is an optimization, not a correctness dependency. A Redis failure disables caching without failing the analysis. The public demo reports both `cache_enabled` and `cache_hit`; a repeated scenario can therefore visibly demonstrate a full-response cache hit.

---

## Concurrency and Failure Handling

- FastAPI keeps its event loop responsive by moving blocking ML, ChromaDB, and orchestration work to worker threads with `asyncio.to_thread`.
- Agents 2 and 3 run in parallel using `ThreadPoolExecutor`.
- PostgreSQL operations use async SQLAlchemy/SQLModel sessions.
- `/batch-analyze` processes batch items sequentially and returns per-item errors; each individual item still uses the parallel specialist stage.
- LLM calls have request timeouts, bounded retries, and Pydantic structured-output validation.
- The complete workflow has an application timeout.
- The public demo returns a typed deterministic fallback when the LLM provider is unavailable; authenticated `/analyze` returns HTTP 503 instead of silently pretending an LLM result succeeded.

---

## Technology Stack

| Layer | Technology |
|-------|------------|
| **LLM** | Configurable Gemini model through Google's OpenAI-compatible API |
| **API** | FastAPI + Pydantic V2 |
| **Orchestration** | Explicit four-agent Python workflow |
| **ML** | XGBoost delivery-time regressor |
| **Knowledge Base** | RAG + persistent ChromaDB vector store |
| **Typed Tool** | Pydantic carrier quote contract + deterministic rate card |
| **Database** | PostgreSQL + SQLModel + async SQLAlchemy |
| **Cache** | Redis LLM cache + predefined demo response cache |
| **Concurrency** | `asyncio.to_thread` + `ThreadPoolExecutor` for Agents 2-3 |
| **Observability** | Structured JSON logs + optional LangSmith tracing |
| **Evaluation** | Deterministic behavioral grading across three specialist outputs |
| **Infrastructure** | Azure VM + Docker Compose + GitHub Actions |
| **Security** | API key header, portfolio Basic Auth, environment-based credentials |

---

## Behavioral Grading

Each specialist output receives an independent score from 0 to 100:

- **Risk:** schema, valid level, score-level alignment, measurable factors, and analysis depth.
- **Carrier:** schema, valid carrier, upgrade/cost consistency, and grounded ROI reasoning.
- **Recovery:** schema, voucher validity, discount consistency, retention estimate consistency, and communication quality.

The combined quality score is not the orchestrator's confidence:

```text
overall_score = (risk_score + carrier_score + recovery_score) / 3
```

For example, specialist scores `80`, `87`, and `80` produce `82.3/100`. Decision confidence remains a separate, bounded estimate returned by Agent 4.

---

## API Endpoints

| Method | Endpoint | Auth | Description |
|--------|----------|------|-------------|
| `GET` | `/` | Basic Auth | Portfolio page and live scenarios |
| `POST` | `/analyze` | `x-api-key` | Full analysis with optional automatic ML/RAG preparation |
| `POST` | `/batch-analyze` | `x-api-key` | Sequential batch with per-item results |
| `POST` | `/predict` | `x-api-key` | XGBoost delivery-time prediction |
| `POST` | `/demo/analyze` | None | Three predefined public portfolio scenarios |
| `GET` | `/health` | None | Live LLM configuration, Redis, PostgreSQL, and ML readiness |
| `GET` | `/status` | None | In-process runtime metrics |
| `GET` | `/debug/llm-test` | `x-api-key` | Authenticated provider connectivity probe |
| `GET` | `/docs` | None | Swagger UI |

### Abridged demo response

```json
{
  "scenario": "moderate",
  "ml_prediction": {
    "predicted_days": 1.5,
    "ml_model_used": true
  },
  "decision": {
    "risk_assessment": {
      "risk_level": "MINIMAL",
      "risk_score": 0.0,
      "primary_risk_factors": [
        "No documented scoring rule triggered"
      ]
    },
    "carrier_recommendation": {
      "recommended_carrier": "Standard Shipping",
      "should_upgrade": false
    },
    "recovery_plan": {
      "voucher_code": null,
      "discount_percentage": 0.0
    },
    "confidence_score": 90.0
  },
  "grading": {
    "overall_score": 82.3,
    "quality_level": "Good",
    "risk_grading": {"score": 80},
    "carrier_grading": {"score": 87},
    "recovery_grading": {"score": 80}
  },
  "cache_enabled": true,
  "cache_hit": false,
  "processing_time_ms": 6300.0
}
```

Exact LLM wording and latency are non-deterministic. The business-rule score, typed tool values, schema, and grading calculation are deterministic.

---

## ML and RAG Context

The delivery-time model was trained on the **Brazilian E-Commerce (Olist) dataset** (~100k orders, 2016-2018). ChromaDB retrieves context from six local documents covering carrier rules, customer recovery, distance, payment lag, weekends/holidays, and weight.

Known limitations:

- The model does not use live weather, traffic, fleet availability, or holiday surge data.
- The typed carrier tool uses a deterministic portfolio rate card, not a live carrier API.
- The tool is a Pydantic-based application contract, not an MCP server. It could be exposed through MCP without changing the agent-facing input/output model.
- The project uses explicit orchestration rather than LangGraph, CrewAI, or AutoGen.
- `rate_limit_check` is currently a permissive extension point, not production rate limiting.

---

## Local Development

```bash
git clone https://github.com/GabrielWalak/Agentic-Logistics-Optimizer.git
cd Agentic-Logistics-Optimizer

cp .env.example .env
# Set LLM_API_KEY and change the example passwords/secrets.

docker compose up -d --build
curl http://localhost:8000/health
```

The LLM provider and model are configured with `LLM_API_KEY`, `LLM_BASE_URL`, and `LLM_MODEL`. The repository defaults target Gemini's OpenAI-compatible endpoint; no GitHub Models token is required.

To run tests in an installed Python environment:

```bash
python -m pytest -q
```

The current suite contains 44 tests covering API behavior, deterministic grading, typed tools, LLM retry/structured output handling, and risk-grounding regressions.

---

## CI/CD and Azure Deployment

GitHub Actions performs:

1. targeted `flake8` runtime-safety checks;
2. the complete pytest suite with PostgreSQL and Redis services;
3. a Buildx image build and entrypoint validation;
4. production deployment only after a successful push to `main`.

The deploy job creates an archive from the exact tested Git revision, transfers it to the Azure VM over SSH, builds an immutable image tagged with the commit SHA, starts it with Docker Compose, and verifies `/health`. If the new application does not become healthy, the script attempts to roll back to the previous release. No Azure Container Registry is required.

See [DEPLOYMENT.md](DEPLOYMENT.md) for the one-time Azure and GitHub configuration.

---

## Project Structure

```text
main.py                       FastAPI routes, state, grading, and demo cache
pydantic_agents.py            Typed models, LLM client, Redis, and orchestration
prompt_engineering.py         Canonical prompts and ResponseGrader
carrier_tools.py              Typed deterministic carrier quote tool
ml_predictor.py               XGBoost inference
logistics_knowledge_base.py   RAG document loading and retrieval
chroma_db_manager.py          Persistent ChromaDB vector store
models.py                     SQLModel audit table
database.py                   Async PostgreSQL engine and health checks
templates/portfolio.py        Server-rendered portfolio and live result UI
tests/                        Unit, integration, and regression tests
deploy/                       Azure production Compose and rollback deployment
.github/workflows/ci.yml      CI, image validation, and Azure deployment
```
