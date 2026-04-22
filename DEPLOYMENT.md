# AgenticAI Logistics System - Deployment Guide

Multi-agent logistics system powered by GitHub Models API, Pydantic, and Docker.

## 🚀 Quick Deploy (Choose Your Path)

### 1️⃣ **Quick Start** (~5 minutes)
For immediate AWS deployment:
→ Read: **[QUICK_START_AWS.md](QUICK_START_AWS.md)**

### 2️⃣ **Detailed Guide** (~20 minutes)
For step-by-step AWS setup:
→ Read: **[AWS_DEPLOYMENT.md](AWS_DEPLOYMENT.md)**

### 3️⃣ **Local Testing** (~10 minutes)
For local Docker testing before AWS:
→ Read: **[BUILD_INSTRUCTIONS.md](BUILD_INSTRUCTIONS.md)**

---

## 📋 What's Inside

```
AgenticAI/
├── app.py                      # Main interactive CLI
├── pydantic_agents.py          # 4-agent orchestration system
├── scenarios_examples.py       # 3 test scenarios (HIGH/LOW/MODERATE risk)
├── prompt_engineering.py       # Optimized prompts + ResponseGrader
├── chroma_db_manager.py        # Vector DB (optional, for RAG)
│
├── Dockerfile                  # Multi-stage build (600-700MB)
├── docker-compose.yml          # Local testing config
├── requirements.txt            # Essential deps only
├── run.sh                       # Container entry point
│
├── BUILD_INSTRUCTIONS.md       # Local Docker build guide
├── AWS_DEPLOYMENT.md           # Detailed AWS setup
├── QUICK_START_AWS.md          # 5-minute quick deploy
├── pre_deploy_check.py         # Pre-deployment validation
│
├── logistics_docs/             # Knowledge base (weight, distance rules)
└── .env.example                # Config template
```

---

## 🎯 System Architecture

```
User Input
    ↓
Interactive CLI (app.py)
    ↓
Multi-Agent Orchestration (pydantic_agents.py)
    ├─ Risk Assessment Agent    (LLM scoring)
    ├─ Carrier Optimization     (shipping decision)
    ├─ Recovery Strategy        (customer retention)
    └─ Decision Integrator      (final recommendation)
    ↓
Response Grading (prompt_engineering.py)
    ├─ Score alignment (0-100)
    ├─ Logic validation
    └─ Specificity checks
    ↓
Output (formatted results)
```

---

## 💡 Features

| Feature | Status |
|---------|--------|
| **Multi-agent system** | ✅ 4 agents (parallel execution) |
| **GitHub Models API** | ✅ gpt-4o-mini (free tier) |
| **Response grading** | ✅ 0-100 scale with deductions |
| **Test scenarios** | ✅ 3 scenarios (HIGH/LOW/MODERATE) |
| **Docker deployment** | ✅ 1GB RAM optimized |
| **AWS compatible** | ✅ t3.micro free tier |
| **FastAPI API** | ⏳ Coming soon |
| **Redis caching** | ⏳ Optional (disabled for t3.micro) |

---

## 📊 Performance

| Metric | Value |
|--------|-------|
| **Analysis time** | ~10-20 seconds per scenario |
| **Docker image** | ~600-700 MB |
| **Memory usage** | ~300-400 MB baseline |
| **Free RAM (t3.micro)** | ~600 MB available ✓ |
| **API calls** | GitHub Models (50/day free) |
| **AWS cost** | **$0/month** (free tier) |

---

## 🔐 Security

- ✅ Non-root user in container
- ✅ .env for secrets (not in image)
- ✅ Read-only logistics_docs volume
- ✅ HTTPS for GitHub Models API
- ✅ No hardcoded credentials

---

## 📚 Test It Locally First

### 1. Setup
```bash
cp .env.example .env
# Edit .env, add GITHUB_TOKEN
nano .env
```

### 2. Build
```bash
docker build -t agentic-ai-logistics:latest .
```

### 3. Run
```bash
docker-compose up
```

### 4. Test Scenarios
```bash
docker exec -it agentic-ai-logistics python app.py
# Select option 2 (all scenarios)
```

---

## 🚀 Deploy to AWS

### Minimum Requirements
- AWS Account (free tier eligible)
- EC2 t3.micro (750 hrs free/month)
- 20GB storage (30GB free)
- GitHub Personal Access Token

### Steps
1. Read **QUICK_START_AWS.md** (5 min)
2. Create EC2 instance (Ubuntu 24.04)
3. SSH into instance
4. Run deployment script
5. Done! ✅

---

## 🧪 Validate Setup

Before deployment, run:
```bash
python pre_deploy_check.py
```

Output:
```
✅ Dockerfile: Dockerfile
✅ requirements.txt: requirements.txt
✅ GITHUB_TOKEN: ghp_XXXXXXXXX...
✅ All checks passed! Ready for deployment!
```

---

## 📝 Configuration

### Environment Variables (.env)
```env
GITHUB_TOKEN=ghp_your_token_here
GITHUB_MODELS_BASE_URL=https://models.inference.ai.azure.com
GITHUB_MODEL=gpt-4o-mini
LLM_TEMPERATURE=0.7
LLM_TOP_P=0.9
LLM_MAX_TOKENS=1000
REDIS_ENABLED=false
```

### Optimization for t3.micro
- ✅ Redis disabled (saves 100+ MB)
- ✅ Python 3.11-slim (minimal image)
- ✅ Single worker process
- ✅ Request timeouts configured
- ✅ Memory limits set

---

## 🆘 Troubleshooting

### Docker build fails
```bash
docker build --no-cache -t agentic-ai-logistics:latest .
```

### Container won't start
```bash
docker logs agentic-app
docker inspect agentic-app
```

### GitHub Models API connection failed
```bash
# Check token
docker exec agentic-app python -c "import os; print(os.getenv('GITHUB_TOKEN'))"

# Test endpoint
docker exec agentic-app curl -I https://models.inference.ai.azure.com
```

### Out of memory (EC2)
```bash
# Add swap
sudo fallocate -l 1G /swapfile
sudo chmod 600 /swapfile
sudo mkswap /swapfile
sudo swapon /swapfile
```

---

## 📖 Documentation

| File | Purpose |
|------|---------|
| **QUICK_START_AWS.md** | 5-minute quick deploy |
| **AWS_DEPLOYMENT.md** | Detailed step-by-step guide |
| **BUILD_INSTRUCTIONS.md** | Local Docker build |
| **pre_deploy_check.py** | Validation script |
| **README.md** | This file (overview) |

---

## 🎓 Learning Resources

### Prompt Engineering
See: `prompt_engineering.py` → RISK_AGENT_PROMPT_V2, CARRIER_AGENT_PROMPT_V2, RECOVERY_AGENT_PROMPT_V2

Optimizations:
- ✅ Structured JSON output
- ✅ Numerical scoring logic
- ✅ Clear decision criteria
- ✅ Quantified metrics (ROI, retention %)

### Response Grading
See: `prompt_engineering.py` → ResponseGrader class

Scoring dimensions:
- Structure (20-25 pts): Required fields present
- Logic (20-25 pts): Values align with rules
- Specificity (20 pts): Concrete measurements
- Depth (15-20 pts): Detailed analysis

---

## 🔄 CI/CD Pipeline (Optional)

Future: GitHub Actions to:
1. Build Docker image
2. Run tests
3. Push to ECR
4. Deploy to EC2

---

## 📞 Support

- **Local issues**: See BUILD_INSTRUCTIONS.md
- **AWS issues**: See AWS_DEPLOYMENT.md
- **Quick help**: See QUICK_START_AWS.md

---

## 📄 License

This project uses GitHub Models API (free tier, 50 calls/day).

---

**Ready to deploy? Start here:** [QUICK_START_AWS.md](QUICK_START_AWS.md) 🚀
