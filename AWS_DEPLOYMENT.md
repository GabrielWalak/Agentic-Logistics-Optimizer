# AWS EC2 t3.micro Deployment Guide

## Prerequisites
- AWS Account (free tier eligible)
- GitHub Personal Access Token (already have it)
- SSH key pair for EC2
- Docker image built locally: `agentic-ai-logistics:latest`

## Step 1: Create EC2 Instance

### 1.1 Launch Instance
```bash
# Go to AWS EC2 Console
# https://console.aws.amazon.com/ec2/
```

**Configuration:**
- **Name**: agentic-ai-logistics
- **AMI**: Ubuntu 24.04 LTS (free tier eligible)
- **Instance Type**: t3.micro ✅
- **Key Pair**: Create new or use existing
- **Storage**: 20 GB gp2 (free tier: up to 30GB)
- **VPC**: Default
- **Public IP**: Enable auto-assign

### 1.2 Security Group (Inbound Rules)
```
SSH:     22    (from your IP)
Custom:  8000  (from 0.0.0.0/0)  <- Optional for FastAPI later
```

### 1.3 Launch & Wait
- Instance status: **running**
- Instance state checks: **2/2 passed**
- Note down: **Public IPv4 address** (e.g., `54.123.45.67`)

---

## Step 2: SSH Into Instance

```bash
# Set permissions on key file (Windows)
icacls "C:\path\to\key.pem" /grant:r "%username%:(F)"
icacls "C:\path\to\key.pem" /inheritance:r

# SSH connection
ssh -i C:\path\to\key.pem ubuntu@54.123.45.67

# Or with WSL
ssh -i ~/key.pem ubuntu@54.123.45.67
```

**Troubleshooting:**
- `Permission denied`: Fix key permissions (see above)
- `Connection refused`: Wait 2-3 min for instance to fully boot
- `Could not resolve hostname`: Check instance IP address

---

## Step 3: Install Docker

Run on EC2 instance:

```bash
# Update packages
sudo apt update
sudo apt upgrade -y

# Install Docker
sudo apt install -y docker.io

# Add ubuntu to docker group (no sudo needed)
sudo usermod -aG docker ubuntu

# Log out and back in
exit
ssh -i C:\path\to\key.pem ubuntu@54.123.45.67

# Verify installation
docker --version
```

---

## Step 4: Clone Repository or Upload Code

### Option A: Git Clone (recommended)
```bash
cd ~
git clone https://github.com/YOUR_GITHUB/AgenticAI.git agentic-ai
cd agentic-ai
```

### Option B: Upload Files
```bash
# From your local machine (PowerShell)
scp -i C:\path\to\key.pem -r f:\python\AgenticAI\* ubuntu@54.123.45.67:~/agentic-ai/
```

---

## Step 5: Setup Environment

On EC2 instance:

```bash
cd ~/agentic-ai

# Create .env file
cat > .env << 'EOF'
GITHUB_TOKEN=your_github_token_here
GITHUB_MODELS_BASE_URL=https://models.inference.ai.azure.com
GITHUB_MODEL=gpt-4o-mini
LLM_TEMPERATURE=0.7
LLM_TOP_P=0.9
LLM_MAX_TOKENS=1000
REDIS_ENABLED=false
PYTHONUNBUFFERED=1
EOF

# Edit with your GitHub token
nano .env
# Paste token, then Ctrl+X, Y, Enter
```

---

## Step 6: Build & Run Docker Image

On EC2 instance:

```bash
# Build image (~2-3 min first time)
docker build -t agentic-ai-logistics:latest .

# Verify build succeeded
docker images | grep agentic-ai

# Run container
docker run -d \
  --name agentic-app \
  --env-file .env \
  -v $(pwd)/logistics_docs:/app/logistics_docs:ro \
  -v $(pwd)/chroma_db:/app/chroma_db \
  --restart unless-stopped \
  agentic-ai-logistics:latest

# Check if running
docker ps
docker logs agentic-app
```

**Expected output in logs:**
```
✓ GITHUB_TOKEN detected
✓ GitHub Models API: https://models.inference.ai.azure.com
✓ Model: gpt-4o-mini
✓ Multi-agent analysis complete
✓ GitHub Models connected. Model: gpt-4o-mini
System Status: READY
```

---

## Step 7: Test Container

### Test 1: Check connection
```bash
docker exec agentic-app python -c "
from pydantic_agents import check_ollama_status
check_ollama_status()
"
```

**Expected:**
```
✓ GitHub Models connected. Model: gpt-4o-mini
```

### Test 2: Run scenarios
```bash
docker exec agentic-app python -c "
from scenarios_examples import SCENARIO_1_HIGH_RISK
from pydantic_agents import run_multi_agent_analysis_parallel
result = run_multi_agent_analysis_parallel(SCENARIO_1_HIGH_RISK)
print('✓ Scenario test passed!')
print(f'Risk Level: {result.risk_assessment.risk_level}')
"
```

---

## Step 8: Verify Resource Usage

```bash
# Check memory usage
docker stats --no-stream agentic-app

# Should show:
# MEM USAGE / LIMIT
# ~300-400M / 1.84G  ✓ Good!

# Check disk space
df -h

# Should have space for logs, etc.
```

---

## Step 9: Monitor Container

### View Logs
```bash
# Real-time logs
docker logs -f agentic-app

# Last 50 lines
docker logs --tail 50 agentic-app

# Since specific time
docker logs --since 10m agentic-app
```

### Restart if needed
```bash
docker restart agentic-app
docker stop agentic-app
docker start agentic-app
```

---

## Step 10: Keep Container Running (Persistent)

### Option A: Auto-restart (already set)
```bash
# Already configured with --restart unless-stopped
# Container auto-starts after reboot
docker run -d --restart unless-stopped ...
```

### Option B: Systemd Service (optional)

Create `/home/ubuntu/docker-run.sh`:
```bash
#!/bin/bash
cd /home/ubuntu/agentic-ai
docker run -d \
  --name agentic-app \
  --env-file .env \
  --restart unless-stopped \
  agentic-ai-logistics:latest
```

Make executable:
```bash
chmod +x /home/ubuntu/docker-run.sh
```

---

## Troubleshooting

### Container won't start
```bash
# Check logs
docker logs agentic-app

# Rebuild without cache
docker build --no-cache -t agentic-ai-logistics:latest .

# Check for port conflicts
docker ps
```

### GitHub Models API connection failed
```bash
# Check environment variables
docker exec agentic-app env | grep GITHUB

# Verify token format
docker exec agentic-app python -c "import os; print(os.getenv('GITHUB_TOKEN')[:20])"

# Test endpoint directly
docker exec agentic-app curl -I https://models.inference.ai.azure.com
```

### Out of memory errors
```bash
# Check memory
free -h
docker stats

# If needed, add swap
sudo fallocate -l 1G /swapfile
sudo chmod 600 /swapfile
sudo mkswap /swapfile
sudo swapon /swapfile

# Make permanent
echo '/swapfile none swap sw 0 0' | sudo tee -a /etc/fstab
```

### SSH connection issues
```bash
# Check security group allows SSH (port 22)
# Check key file permissions
ls -la ~/.ssh/key.pem

# If wrong permissions:
chmod 600 ~/.ssh/key.pem
```

---

## Production Checklist

- [ ] EC2 instance running (t3.micro)
- [ ] Security group configured (SSH + optional 8000)
- [ ] Docker installed and working
- [ ] Code cloned/uploaded to EC2
- [ ] `.env` file created with GITHUB_TOKEN
- [ ] Docker image built successfully
- [ ] Container running (`docker ps`)
- [ ] GitHub Models API connected (check logs)
- [ ] Test scenario executed successfully
- [ ] Memory usage < 500 MB
- [ ] Auto-restart configured
- [ ] Logs monitored for errors

---

## Cost Estimation (AWS Free Tier)

| Service | Free Tier | Usage |
|---------|-----------|-------|
| EC2 t3.micro | 750 hrs/month | 730 hrs/month ✓ |
| Data transfer | 100 GB/month out | ~1-5 GB/month ✓ |
| EBS storage | 30 GB/month | 20 GB ✓ |
| **Total Cost** | **$0/month** | ✅ |

**First 12 months**: Completely free!

---

## Next Steps

### Option A: Add FastAPI Later
```bash
# When ready:
pip install fastapi uvicorn
# Create api.py wrapper
# Deploy with Uvicorn on port 8000
```

### Option B: Upgrade Instance
```bash
# If performance issues:
# Stop instance → Change instance type → t3.small/t3.medium → Start
# Still free tier eligible until 12 months
```

### Option C: Setup Auto-Scaling
```bash
# For production:
# Use AWS Auto Scaling Group
# ALB (Application Load Balancer)
# CloudWatch monitoring
```

---

## Useful AWS Commands

```bash
# SSH quick access alias (add to .bashrc/.zshrc)
alias ssh-agentic='ssh -i ~/path/to/key.pem ubuntu@YOUR_IP'

# Monitor from local machine
ssh -i ~/key.pem ubuntu@IP 'docker logs -f agentic-app'

# Update code and restart
ssh -i ~/key.pem ubuntu@IP << 'SCRIPT'
cd ~/agentic-ai
git pull
docker build -t agentic-ai-logistics:latest .
docker restart agentic-app
SCRIPT
```

---

## Support & Cleanup

### If something goes wrong
1. Check logs: `docker logs agentic-app`
2. Check resources: `docker stats`
3. Restart: `docker restart agentic-app`
4. Rebuild: `docker build --no-cache ...`

### To delete instance (cleanup)
```bash
# AWS Console → EC2 → Instances → Right-click → Terminate
# Or via AWS CLI:
aws ec2 terminate-instances --instance-ids i-1234567890abcdef0
```

---

**🎉 Ready? Go to Step 1: Create EC2 Instance**
