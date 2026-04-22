# Docker Build & Deployment Guide for t3.micro

## Prerequisites
- Docker installed
- GitHub Personal Access Token (with `gist` or `repo` permissions)
- AWS EC2 t3.micro instance with Docker

## Local Testing

### 1. Setup Environment
```bash
cp .env.example .env
# Edit .env and add your GITHUB_TOKEN
```

### 2. Build Docker Image
```bash
docker build -t agentic-ai-logistics:latest .
```

Image size: **~600-700 MB** (optimized for t3.micro)

### 3. Run Locally
```bash
docker-compose up -d
docker-compose logs -f
```

Test with:
```bash
docker exec -it agentic-ai-logistics python -c "
from pydantic_agents import check_ollama_status
if check_ollama_status():
    print('✓ GitHub Models connected!')
"
```

### 4. Interactive Menu
```bash
docker exec -it agentic-ai-logistics python app.py
```

## AWS EC2 t3.micro Deployment

### 1. Launch EC2 Instance
- **AMI**: Ubuntu 24.04 LTS (free tier eligible)
- **Instance Type**: t3.micro
- **Storage**: 20GB gp2
- **Security Group**: Allow SSH (22), HTTP (80), HTTPS (443)

### 2. SSH Into Instance
```bash
ssh -i your-key.pem ubuntu@<instance-ip>
```

### 3. Install Docker
```bash
sudo apt update
sudo apt install -y docker.io
sudo usermod -aG docker ubuntu
# Log out and back in
```

### 4. Clone/Upload Code
```bash
git clone <your-repo> agentic-ai
cd agentic-ai
cp .env.example .env
# Edit .env with GITHUB_TOKEN
nano .env
```

### 5. Build & Run
```bash
docker build -t agentic-ai-logistics:latest .
docker run -d \
  --name agentic-app \
  --env-file .env \
  -v $(pwd)/logistics_docs:/app/logistics_docs:ro \
  agentic-ai-logistics:latest
```

### 6. Check Logs
```bash
docker logs -f agentic-app
```

### 7. Test Connection
```bash
docker exec agentic-app python -c "from pydantic_agents import check_ollama_status; check_ollama_status()"
```

## Resource Optimization for t3.micro (1GB RAM)

### Memory Profile
- Docker overhead: ~100-150 MB
- Python + dependencies: ~150-200 MB
- App runtime (per request): ~100-150 MB
- **Total baseline**: ~400 MB
- **Free for requests**: ~600 MB ✓

### Performance Tips
1. **Disable Redis** - Already configured (REDIS_ENABLED=false)
2. **Single worker** - App runs as single process
3. **Request timeout** - Set to 30 seconds (GitHub Models API)
4. **Monitor memory** - Check with `docker stats`

## Troubleshooting

### Out of Memory Errors
```bash
# Check memory usage
docker stats --no-stream

# If OOM, reduce processes or enable swap
sudo fallocate -l 1G /swapfile
sudo chmod 600 /swapfile
sudo mkswap /swapfile
sudo swapon /swapfile
```

### GitHub Models API Connection Failed
1. Verify GITHUB_TOKEN is set: `echo $GITHUB_TOKEN`
2. Test endpoint: `curl -I https://models.inference.ai.azure.com`
3. Check .env file is loaded: `docker exec agentic-app env | grep GITHUB`

### Container Won't Start
```bash
# Check logs
docker logs agentic-app

# Rebuild with no cache
docker build --no-cache -t agentic-ai-logistics:latest .
```

## Production Checklist

- [ ] GITHUB_TOKEN set securely in .env (not in Dockerfile)
- [ ] Redis disabled for t3.micro
- [ ] Tested locally with docker-compose
- [ ] Image size verified (~600-700 MB)
- [ ] EC2 instance has enough storage for Docker
- [ ] Security group allows necessary ports
- [ ] GitHub Models API rate limits checked (50 calls/day free tier)
- [ ] Logs configured for monitoring
- [ ] Restart policy set (--restart unless-stopped)

## Cleanup

### Local
```bash
docker-compose down
docker image rm agentic-ai-logistics:latest
```

### AWS EC2
```bash
docker stop agentic-app
docker rm agentic-app
docker image rm agentic-ai-logistics:latest
```

## Monitoring

### View Logs
```bash
docker logs agentic-app
docker logs -f agentic-app  # Follow mode
```

### Check Resource Usage
```bash
docker stats agentic-app
```

### Keep Container Running
If using SSH and need persistent process:
```bash
# Run with nohup in background
nohup docker run -d \
  --restart unless-stopped \
  --name agentic-app \
  --env-file .env \
  agentic-ai-logistics:latest &
```

Or use systemd service (see EC2 best practices).
