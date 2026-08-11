# Runtime Architecture Flow

This document describes the behavior implemented by the current code. The
workflow uses explicit Python orchestration: LLMs propose typed specialist
outputs, while deterministic code owns risk scoring, carrier quotes, fallback
selection, grading, timeouts, and persistence decisions.

## System overview

```mermaid
flowchart LR
    Client[Client or portfolio UI]

    subgraph API[FastAPI application]
        Routes[HTTP routes]
        Prepare[Scenario preparation]
        Workflow[Four-agent workflow]
        Grade[Behavioral grader]
        Fallback[Deterministic demo fallback]
    end

    subgraph Compute[Local computation]
        ML[XGBoost model]
        CarrierTool[Typed carrier quote tool]
        Rules[Risk and validation rules]
    end

    subgraph Providers[External provider]
        Gemini[Gemini OpenAI-compatible API]
    end

    subgraph State[State and storage]
        Redis[(Redis)]
        Postgres[(PostgreSQL)]
        Chroma[(Persistent ChromaDB)]
        Metrics[In-process AppState]
    end

    Client --> Routes
    Routes --> Prepare
    Prepare --> ML
    Prepare --> Chroma
    Prepare --> Workflow
    Workflow --> Gemini
    Workflow --> CarrierTool
    Workflow --> Rules
    Workflow --> Redis
    Workflow --> Grade
    Workflow -. LLM unavailable in demo .-> Fallback
    Grade --> Postgres
    Routes --> Metrics
    Grade --> Client
    Fallback --> Client
```

Redis is an optimization rather than a correctness dependency. ChromaDB is an
enrichment layer and falls back to neutral context when retrieval fails.
PostgreSQL audit persistence is best effort and applies only to successful
authenticated `/analyze` requests.

## Endpoint entry paths

```mermaid
flowchart TD
    Request[Incoming HTTP request] --> Route{Endpoint}

    Route -->|/analyze| AnalyzeAuth[x-api-key verification]
    Route -->|/batch-analyze| BatchAuth[x-api-key verification]
    Route -->|/demo/analyze| DemoInput[Predefined scenario validation]
    Route -->|/predict| MLOnly[XGBoost prediction only]

    AnalyzeAuth --> AnalyzePrepare[Prepare one scenario]
    BatchAuth --> BatchLoop[Process items sequentially]
    BatchLoop --> BatchPrepare[Prepare current scenario]

    DemoInput --> DemoCache{Completed response in Redis?}
    DemoCache -->|yes| DemoHit[Return cached payload]
    DemoCache -->|no| DemoPrepare[Run ML and RAG preparation]

    AnalyzePrepare --> AgentWorkflow[Agent workflow]
    BatchPrepare --> AgentWorkflow
    DemoPrepare --> AgentWorkflow
```

`rate_limit_check` is currently a permissive extension hook; it does not yet
enforce a production rate limit. The portfolio page at `/` uses Basic Auth,
while the predefined demo-analysis endpoint itself is public.

## Scenario preparation: ML and RAG

`/analyze` and each `/batch-analyze` item use
`_prepare_analysis_scenario`. The demo follows the same logical preparation
with fixed server-side inputs.

```mermaid
sequenceDiagram
    autonumber
    participant API as FastAPI endpoint
    participant Worker as asyncio worker thread
    participant ML as XGBoost predictor
    participant Vector as ChromaDB
    participant KB as Logistics documents

    API->>Worker: prepare AnalysisRequest

    alt predicted_days is missing or non-positive
        Worker->>ML: predict_delivery_days(features)
        ML-->>Worker: predicted days or None
        opt model returns None
            Worker->>Worker: use 7.0-day fallback
        end
    else caller supplied predicted_days
        Worker->>Worker: keep supplied prediction
    end

    alt rag_context has the default marker
        Worker->>Vector: open persistent collection
        alt collection is empty
            Vector->>KB: index six source documents
        end
        Worker->>Vector: similarity query for scenario
        Vector-->>Worker: relevant context
        opt retrieval fails or returns no context
            Worker->>Worker: use neutral default context
        end
    else caller supplied custom context
        Worker->>Worker: keep supplied context
    end

    Worker-->>API: validated DeliveryScenario
```

Blocking ML and vector operations run through `asyncio.to_thread`, keeping the
FastAPI event loop available for other requests.

## Agent orchestration and concurrency

```mermaid
flowchart TD
    Scenario[DeliveryScenario] --> RiskRules[Calculate authoritative risk score]
    RiskRules --> RiskLLM[Agent 1: explain risk]
    RiskLLM --> NormalizeRisk[Replace contradictory or unsupported claims]

    NormalizeRisk --> Parallel{ThreadPoolExecutor max_workers=2}

    Parallel --> CarrierStart[Agent 2: carrier optimization]
    CarrierStart --> AllQuotes[get_all_carrier_quotes]
    AllQuotes --> QuoteTool[get_carrier_quote for each of 5 carriers]
    QuoteTool --> RateCard[Local deterministic rate card]
    RateCard --> CarrierLLM[LLM selects and explains an available option]
    CarrierLLM --> SelectQuote[select_carrier_quote validates or falls back]
    SelectQuote --> CarrierResult[CarrierRecommendation]

    Parallel --> RecoveryLLM[Agent 3: recovery strategy]
    RecoveryLLM --> RecoveryRules[Normalize voucher, discount and retention]
    RecoveryRules --> RecoveryResult[CustomerRecoveryPlan]

    CarrierResult --> Join[Wait for both futures]
    RecoveryResult --> Join
    Join --> Orchestrator[Agent 4: integrate specialist outputs]
    Orchestrator --> Bound[Bound confidence and validate summary]
    Bound --> Decision[IntegratedDecision]
```

Agent 1 must finish first because its normalized risk output is an input to
Agents 2 and 3. Only Agents 2 and 3 execute concurrently. Agent 4 starts after
both futures have completed.

The carrier capability is deliberately split into three responsibilities:

1. `get_carrier_quote` is the core typed deterministic tool.
2. `get_all_carrier_quotes` calls it for every local carrier profile.
3. `select_carrier_quote` validates the LLM recommendation and chooses the
   fastest or cheapest available fallback when necessary.

The tool does not call a production carrier API. Its Pydantic contracts allow
the local rate card to be replaced later by an authenticated REST or MCP
implementation without changing the agent-facing result shape.

## One LLM call and its Redis cache

Every agent calls the same provider-neutral `call_ollama` function. The legacy
name is retained for compatibility; the configured implementation uses an
OpenAI-compatible client and currently targets Gemini.

```mermaid
sequenceDiagram
    autonumber
    participant Agent
    participant Cache as Redis LLM cache
    participant Client as OpenAI-compatible client
    participant LLM as Gemini API
    participant Schema as Pydantic model

    Agent->>Agent: build key from model, sampling, schema and prompts
    Agent->>Cache: GET cache key

    alt cache hit
        Cache-->>Agent: structured JSON
    else cache miss or Redis unavailable
        loop up to configured retry limit
            Agent->>Client: system prompt + user prompt + response schema
            Client->>LLM: HTTPS request
            LLM-->>Client: structured response or provider error
        end
        Client->>Schema: validate parsed output
        Schema-->>Agent: typed result
        Agent->>Cache: SETEX result with 1-hour TTL
    end
```

Authentication, invalid-request, retired-endpoint, and quota errors are treated
as permanent for the current request and are not repeatedly retried. Redis
connection or command failures only disable the cache path.

## Grading, persistence, and response behavior

```mermaid
flowchart TD
    Decision[IntegratedDecision] --> Route{Calling endpoint}

    Route -->|/analyze| GradeAnalyze[Grade risk, carrier and recovery]
    GradeAnalyze --> MeanAnalyze[Arithmetic mean: overall_score]
    MeanAnalyze --> DBReady{PostgreSQL ready?}
    DBReady -->|yes| Audit[Best-effort AuditLog commit]
    DBReady -->|no| AnalyzeResponse[AnalysisResponse]
    Audit --> AnalyzeResponse

    Route -->|/demo/analyze| GradeDemo[Grade risk, carrier and recovery]
    GradeDemo --> MeanDemo[Arithmetic mean: overall_score]
    MeanDemo --> DemoWrite[Cache completed payload for 10 minutes]
    DemoWrite --> DemoResponse[Demo JSON response]

    Route -->|/batch-analyze| BatchResult[Append decision to current item]
    BatchResult --> Next{More items?}
    Next -->|yes| PrepareNext[Prepare next item]
    Next -->|no| BatchResponse[Batch summary and per-item results]
```

The grader's `overall_score` and Agent 4's `confidence_score` are separate
values. The former is the arithmetic mean of three deterministic specialist
scores; the latter is a bounded evidence-quality estimate in the decision.

## Timeouts and failure paths

```mermaid
flowchart TD
    Run[Run workflow through asyncio.wait_for] --> Outcome{Outcome}

    Outcome -->|success| Success[Return endpoint-specific response]
    Outcome -->|timeout| Endpoint{Endpoint}
    Outcome -->|LLMError| LLMEndpoint{Endpoint}
    Outcome -->|other exception| Other[HTTP 500 or batch item error]

    Endpoint -->|/analyze or /demo/analyze| Timeout504[HTTP 504]
    Endpoint -->|/batch-analyze| TimeoutItem[Record timeout for current item]

    LLMEndpoint -->|/analyze| Unavailable503[HTTP 503]
    LLMEndpoint -->|/demo/analyze| DemoFallback[Build deterministic fallback decision]
    LLMEndpoint -->|/batch-analyze| LLMItem[Record error for current item]

    DemoFallback --> FallbackGrade[Run deterministic grader]
    FallbackGrade --> FallbackResponse[Return fallback_used true]
```

The authenticated analysis endpoint never presents a deterministic fallback as
an LLM-generated decision. The public demo explicitly exposes `fallback_used`
and `fallback_reason` so the degraded mode remains visible.

## State ownership

| State | Owner | Lifetime | Role |
|---|---|---|---|
| Request and agent values | Pydantic models | One analysis | Validate boundaries between stages |
| Counters and latency | `AppState` | Process lifetime | `/status` operational metrics |
| Database readiness | `app.state.db_ready` | Refreshed at runtime | Gate best-effort audit writes |
| LLM response cache | Redis | 1-hour TTL | Reuse schema-aware agent responses |
| Demo response cache | Redis | 10-minute TTL | Reuse complete predefined results |
| Audit log | PostgreSQL/SQLModel | Durable | Store successful `/analyze` inputs and decisions |
| Vector collection | Persistent ChromaDB | Durable local volume | Retrieve logistics knowledge context |

## Source map

| Responsibility | Implementation |
|---|---|
| HTTP endpoints, preparation, timeout and grading | `main.py` |
| Agent models, LLM client, Redis and orchestration | `pydantic_agents.py` |
| Carrier quote contracts and deterministic rate card | `carrier_tools.py` |
| XGBoost inference | `ml_predictor.py` |
| ChromaDB indexing and retrieval | `chroma_db_manager.py` |
| PostgreSQL engine and sessions | `database.py` |
| SQLModel audit tables | `models.py` |
| Prompts and deterministic grader | `prompt_engineering.py` |
