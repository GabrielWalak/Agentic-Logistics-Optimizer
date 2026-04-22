# AWS Quick Start (5 minutes)

## TL;DR

```bash
# 1. Create EC2 t3.micro instance (Ubuntu 24.04)
# 2. SSH into instance
ssh -i key.pem ubuntu@YOUR_IP

# 3. Run one command
curl -fsSL https://raw.githubusercontent.com/YOUR_GITHUB/AgenticAI/main/deploy.sh | bash

# 4. Edit .env when prompted with your GITHUB_TOKEN

# 5. Done! ✅
```

---

## Detailed Steps

### Step 1: Create AWS EC2 Instance (3 minutes)

1. Go to [AWS Console](https://console.aws.amazon.com/ec2/)
2. Click **"Launch Instance"**
3. Fill in:
   - **Name**: agentic-ai-logistics
   - **AMI**: Ubuntu 24.04 LTS
   - **Instance Type**: t3.micro ✅
   - **Key Pair**: Create new (save .pem file!)
   - **Security Group**:
     - SSH (22): from your IP
     - Custom (8000): 0.0.0.0/0 (optional)
   - **Storage**: 20GB (default)
4. Click **"Launch Instance"**
5. Wait 1-2 minutes for instance to fully boot
6. Copy **Public IPv4 address** (e.g., `54.123.45.67`)

### Step 2: SSH Into Instance (1 minute)

**Windows (PowerShell):**
```powershell
# Set key permissions
icacls "C:\path\to\key.pem" /grant:r "$($env:USERNAME):(F)"
icacls "C:\path\to\key.pem" /inheritance:r

# SSH
ssh -i C:\path\to\key.pem ubuntu@54.123.45.67
```

**Mac/Linux:**
```bash
chmod 600 ~/Downloads/key.pem
ssh -i ~/Downloads/key.pem ubuntu@54.123.45.67
```

**WSL (Windows):**
```bash
ssh -i ~/key.pem ubuntu@54.123.45.67
```

### Step 3: Deploy (1 minute)

Once SSH'd into instance:

```bash
# Option A: Automated (if you have GitHub repo)
curl -fsSL https://raw.githubusercontent.com/YOUR_GITHUB/AgenticAI/main/deploy.sh | bash

# Option B: Manual
cd ~
git clone https://github.com/YOUR_GITHUB/AgenticAI.git agentic-ai
cd agentic-ai
nano ~/.agentic.env  # Add your GITHUB_TOKEN
docker build -t agentic-ai-logistics:latest .
docker run -d --name agentic-app --env-file ~/.agentic.env --restart unless-stopped agentic-ai-logistics:latest
```

**⚠️ SECURITY NOTE:** Token is stored in `~/.agentic.env` (home directory), NOT in the repository or Docker image!

### Step 4: Verify (30 seconds)

```bash
# Check running
docker ps

# Check logs
docker logs agentic-app

# Test GitHub Models API
docker exec agentic-app python -c "from pydantic_agents import check_ollama_status; check_ollama_status()"
```

**Expected output:**
```
✓ GitHub Models connected. Model: gpt-4o-mini
System Status: READY
```

---

## All Set! 🎉

### What you can do now:

1. **Run scenarios** (inside container):
   ```bash
   docker exec agentic-app python -c "
   from scenarios_examples import run_all_scenarios
   run_all_scenarios()
   "
   ```

2. **Monitor logs** (real-time):
   ```bash
   docker logs -f agentic-app
   ```

3. **SSH back anytime**:
   ```bash
   ssh -i key.pem ubuntu@IP
   docker ps
   ```

---

## Troubleshooting

| Problem | Solution |
|---------|----------|
| `Permission denied` (SSH) | Fix key: `chmod 600 key.pem` |
| `Connection refused` | Wait 2-3 min for EC2 to boot |
| `Docker: command not found` | Logout and back in: `exit` → `ssh` again |
| `GITHUB_TOKEN not found` | Edit `.env` and add token: `nano .env` |
| `Container exits` | Check logs: `docker logs agentic-app` |
| Out of memory | Check: `docker stats` and `free -h` |

---

## Cost: $0 (Free Tier) ✅

- t3.micro: 750 hrs/month (included)
- 20GB storage: Included (up to 30GB free)
- Data transfer: ~1-5GB/month (100GB free)

**First 12 months**: Completely free!

---

## Next Steps

### Later: Add FastAPI API
```bash
# When ready, add HTTP endpoints:
docker exec agentic-app pip install fastapi uvicorn
# Restart container with port mapping
```

### Later: Custom Domain
```bash
# Setup Route53 or CloudFlare DNS
# Point to EC2 Public IP
# Setup HTTPS with Let's Encrypt
```

### Later: Auto-Scaling
```bash
# Use Load Balancer + Auto Scaling Group
# Multiple EC2 instances
# CloudWatch monitoring
```

---

## SSH Aliases (Optional)

Add to `~/.bashrc` or `~/.zshrc`:

```bash
alias ssh-agentic='ssh -i ~/key.pem ubuntu@YOUR_IP'
alias docker-logs='ssh -i ~/key.pem ubuntu@YOUR_IP docker logs -f agentic-app'
alias docker-stats='ssh -i ~/key.pem ubuntu@YOUR_IP docker stats --no-stream'
```

Then just use:
```bash
ssh-agentic
docker-logs
docker-stats
```

---

## Support

For detailed instructions, see: **[AWS_DEPLOYMENT.md](AWS_DEPLOYMENT.md)**

For local testing: **[BUILD_INSTRUCTIONS.md](BUILD_INSTRUCTIONS.md)**

---

**🚀 Ready? Create EC2 instance now → [AWS Console](https://console.aws.amazon.com/ec2/)**
