# 🤖 AgenticAI - Multi-Agent Logistics Decision System

[![Python 3.10+](https://img.shields.io/badge/Python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![Ollama](https://img.shields.io/badge/LLM-Ollama%20Mistral-green.svg)](https://ollama.ai/)
[![ChromaDB](https://img.shields.io/badge/RAG-ChromaDB-orange.svg)](https://www.trychroma.com/)
[![XGBoost](https://img.shields.io/badge/ML-XGBoost-red.svg)](https://xgboost.readthedocs.io/)

**AgenticAI** is an enterprise-grade Decision Intelligence system that combines Machine Learning predictions with a Multi-Agent LLM architecture to autonomously analyze e-commerce delivery risks and initiate customer recovery protocols.

> 🎯 **Mission**: Bridge the gap between raw ML predictions and strategic business action through intelligent agent orchestration.

---

## 📐 System Architecture

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                         OLIST AGENTIC AI SYSTEM                             │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  ┌──────────────┐    ┌──────────────┐    ┌──────────────────────────────┐  │
│  │   INPUT      │    │   XGBOOST    │    │     CHROMADB RAG             │  │
│  │  Delivery    │───▶│   Model      │───▶│  ┌────────────────────────┐  │  │
│  │  Scenario    │    │  (R²=0.41)   │    │  │ Semantic Search        │  │  │
│  └──────────────┘    └──────────────┘    │  │ all-MiniLM-L6-v2       │  │  │
│                             │            │  │ 39 Knowledge Chunks    │  │  │
│                             ▼            │  └────────────────────────┘  │  │
│                      Predicted Days      └──────────────┬───────────────┘  │
│                             │                           │                  │
│                             ▼                           ▼                  │
│  ┌──────────────────────────────────────────────────────────────────────┐  │
│  │                    MULTI-AGENT ORCHESTRATION                         │  │
│  │  ┌─────────────────────────────────────────────────────────────────┐ │  │
│  │  │                    Ollama + Mistral 7B                          │ │  │
│  │  │                   (Local LLM Inference)                         │ │  │
│  │  └─────────────────────────────────────────────────────────────────┘ │  │
│  │                              │                                       │  │
│  │        ┌─────────────────────┼─────────────────────┐                 │  │
│  │        ▼                     ▼                     ▼                 │  │
│  │  ┌───────────┐        ┌───────────┐        ┌───────────┐             │  │
│  │  │   RISK    │        │  CARRIER  │        │ RECOVERY  │             │  │
│  │  │   AGENT   │        │   AGENT   │        │   AGENT   │             │  │
│  │  │           │        │           │        │           │             │  │
│  │  │ • Score   │        │ • Upgrade │        │ • Voucher │             │  │
│  │  │ • Factors │        │ • Cost    │        │ • Timing  │             │  │
│  │  │ • Priority│        │ • ROI     │        │ • Message │             │  │
│  │  └─────┬─────┘        └─────┬─────┘        └─────┬─────┘             │  │
│  │        │                    │                    │                   │  │
│  │        └────────────────────┼────────────────────┘                   │  │
│  │                             ▼                                        │  │
│  │                    ┌─────────────────┐                               │  │
│  │                    │  ORCHESTRATOR   │                               │  │
│  │                    │     AGENT       │                               │  │
│  │                    │                 │                               │  │
│  │                    │ Executive       │                               │  │
│  │                    │ Summary +       │                               │  │
│  │                    │ Confidence      │                               │  │
│  │                    └─────────────────┘                               │  │
│  └──────────────────────────────────────────────────────────────────────┘  │
│                              │                                             │
│                              ▼                                             │
│  ┌──────────────────────────────────────────────────────────────────────┐  │
│  │                      INTEGRATED DECISION                             │  │
│  │  • Risk Level (HIGH/MODERATE/LOW)    • Carrier Recommendation        │  │
│  │  • Voucher Code (DELAY15/DELAY50)    • Customer Communication        │  │
│  │  • ROI Analysis                      • Confidence Score              │  │
│  └──────────────────────────────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## 🚀 Key Features

| Feature | Description |
|---------|-------------|
| **🧠 Multi-Agent Architecture** | 4 specialized AI agents (Risk, Carrier, Recovery, Orchestrator) working in parallel |
| **📊 XGBoost Predictions** | ML model trained on 100k+ Olist orders predicting delivery time |
| **🔍 RAG-Enhanced Context** | ChromaDB vector database with domain knowledge for informed decisions |
| **⚡ Redis Caching** | 100x speedup on repeated queries with intelligent response caching |
| **🔒 Pydantic Validation** | Type-safe outputs with robust LLM response parsing |
| **📈 LangSmith Tracing** | Optional observability and monitoring for production deployments |
| **🖥️ Rich CLI Dashboard** | Beautiful terminal interface with real-time status updates |

---

## 🛠️ Tech Stack

| Category | Technology |
|----------|------------|
| **Language** | Python 3.10+ |
| **LLM** | Ollama + Mistral 7B (local inference) |
| **ML Framework** | XGBoost, Scikit-learn, Pandas |
| **Vector Database** | ChromaDB + Sentence Transformers (all-MiniLM-L6-v2) |
| **Caching** | Redis / Memurai (Windows) |
| **Validation** | Pydantic v2 |
| **Monitoring** | LangSmith (optional) |
| **Interface** | Rich (Interactive CLI) |

---

## 📦 Installation

### Prerequisites
- Python 3.10+
- [Ollama](https://ollama.ai/) with Mistral model
- Redis/Memurai (optional, for caching)

### Setup

```bash
# Clone the repository
git clone https://github.com/yourusername/AgenticAI.git
cd AgenticAI

# Create virtual environment
python -m venv .venv
.venv\Scripts\activate  # Windows
# source .venv/bin/activate  # Linux/Mac

# Install dependencies
pip install -r requirements.txt

# Pull Mistral model for Ollama
ollama pull mistral

# Run the system
python main.py
```

### Dependencies

```
pydantic>=2.0
ollama
chromadb
sentence-transformers
xgboost
scikit-learn
pandas
numpy
rich
python-dotenv
redis  # optional
langsmith  # optional
```

---

## 🎮 Usage

### Running the Full System

```bash
python main.py
```

This will:
1. Load the XGBoost prediction model
2. Initialize ChromaDB with logistics knowledge
3. Run 3 test scenarios through the multi-agent system
4. Display integrated decisions in the CLI dashboard

### Testing Agents Directly

```bash
python pydantic_agents.py
```

Verifies Ollama connection and agent system status.

---

## 📊 Business Scenarios

The system evaluates three distinct delivery scenarios:

### 🔴 Case 1: High Risk Scenario
**Input:** Heavy package (5kg), Long distance (1200km), Peak Season (November)

**Agent Actions:**
- Risk Agent → HIGH (85/100)
- Carrier Agent → Upgrade to Premium Express (+R$45)
- Recovery Agent → Issue DELAY50 voucher proactively

![High Risk Scenario](screenshots/1.png)

---

### 🟢 Case 2: Low Risk Scenario
**Input:** Lightweight package (0.5kg), Local delivery (45km), Off-season

**Agent Actions:**
- Risk Agent → LOW (25/100)
- Carrier Agent → Keep Standard Shipping (cost savings)
- Recovery Agent → No intervention needed

![Low Risk Scenario](screenshots/2.png)

---

### 🟡 Case 3: Medium Risk Scenario
**Input:** Medium package (1.2kg), Payment lag (4 days), Weekend order

**Agent Actions:**
- Risk Agent → MODERATE (55/100)
- Carrier Agent → Evaluate based on margins
- Recovery Agent → Prepare DELAY15 voucher if needed

![Medium Risk Scenario](screenshots/3.png)

---

## 🧩 Project Structure

```
AgenticAI/
├── main.py                    # Entry point - orchestrates full workflow
├── pydantic_agents.py         # Multi-agent system with Pydantic models
├── chroma_db_manager.py       # ChromaDB RAG system manager
├── logistics_knowledge_base.py # Domain knowledge documents
├── xgboost_model.pkl          # Trained ML model (not in repo)
├── chroma_db/                 # Persistent vector database
│   └── chroma.sqlite3
├── logistics_docs/            # Source documents for RAG
│   ├── carrier_rules.txt
│   ├── customer_recovery.txt
│   ├── distance_guidelines.txt
│   ├── payment_lag_impact.txt
│   ├── weekend_holidays.txt
│   └── weight_volume_rules.txt
├── screenshots/               # CLI output examples
│   ├── 1.png
│   ├── 2.png
│   └── 3.png
└── README.md
```

---

## 🔧 Configuration

### Environment Variables (`.env`)

```env
# LangSmith (optional monitoring)
LANGCHAIN_TRACING_V2=true
LANGCHAIN_API_KEY=your_api_key
LANGCHAIN_PROJECT=AgenticAI

# Redis (optional caching)
REDIS_HOST=localhost
REDIS_PORT=6379
```

### Ollama Configuration

The system uses Mistral 7B with optimized parameters:
- **Temperature:** 0.3 (consistent outputs)
- **Max Tokens:** 1024
- **Context Window:** Scenario + RAG chunks

---

## 📈 Performance

| Metric | Value |
|--------|-------|
| XGBoost Model R² | 0.41 |
| Average Agent Response | 8-15 seconds |
| Cached Response | ~0.02 seconds |
| RAG Retrieval | ~0.1 seconds |
| Total Pipeline (uncached) | 30-45 seconds |
| Total Pipeline (cached) | ~2 seconds |

---

## 🔄 How It Works

### 1. Input Processing
```python
scenario = {
    'product_weight_g': 5000,
    'distance_km': 1200.0,
    'payment_lag_days': 2,
    'is_weekend_order': 1,
    'freight_value': 85.00
}
```

### 2. XGBoost Prediction
The ML model predicts delivery time based on 11 features trained on historical Olist data.

### 3. RAG Context Retrieval
ChromaDB performs semantic search to find relevant logistics policies:
- Carrier rules and SLAs
- Weekend/holiday impact guidelines
- Weight and volume restrictions
- Customer recovery protocols

### 4. Multi-Agent Analysis
Four specialized agents analyze the scenario in parallel:
- **Risk Agent**: Calculates risk score and identifies primary factors
- **Carrier Agent**: Recommends carrier upgrades with ROI analysis
- **Recovery Agent**: Designs proactive customer retention strategy
- **Orchestrator**: Synthesizes all inputs into executive summary

### 5. Integrated Decision
Final output includes actionable recommendations with confidence scores.

---

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

---

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

## 🙏 Acknowledgments

- [Olist Dataset](https://www.kaggle.com/olistbr/brazilian-ecommerce) - Brazilian E-commerce data
- [Ollama](https://ollama.ai/) - Local LLM inference
- [ChromaDB](https://www.trychroma.com/) - Vector database
- [Pydantic](https://github.com/pydantic/pydantic) - Data validation

---

<p align="center">
  <strong>Built with 🤖 AI-Powered Decision Intelligence</strong>
</p>
