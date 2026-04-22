#!/bin/bash
# Quick AWS Deployment Script
# Run this on your EC2 instance to deploy in ~5 minutes
# SECURITY: Stores secrets securely, never in Docker image

set -e

echo "=========================================="
echo "AgenticAI Logistics - AWS Quick Deploy"
echo "=========================================="

# Step 1: Update system
echo "[1/6] Updating system..."
sudo apt update -qq
sudo apt upgrade -y -qq

# Step 2: Install Docker
echo "[2/6] Installing Docker..."
sudo apt install -y -qq docker.io
sudo usermod -aG docker ubuntu

# Step 3: Clone repo (or upload manually)
echo "[3/6] Cloning repository..."
cd ~
if [ ! -d "agentic-ai" ]; then
    git clone https://github.com/YOUR_GITHUB/AgenticAI.git agentic-ai || {
        echo "⚠️ Git clone failed. Upload files manually and continue."
        mkdir -p ~/agentic-ai
    }
fi
cd agentic-ai

# Step 4: Create secure .env in HOME (NOT in repo)
echo "[4/6] Setting up secure environment..."
if [ ! -f ~/.agentic.env ]; then
    cat > ~/.agentic.env << 'EOF'
GITHUB_TOKEN=YOUR_TOKEN_HERE
GITHUB_MODELS_BASE_URL=https://models.inference.ai.azure.com
GITHUB_MODEL=gpt-4o-mini
LLM_TEMPERATURE=0.7
LLM_TOP_P=0.9
LLM_MAX_TOKENS=1000
REDIS_ENABLED=false
PYTHONUNBUFFERED=1
EOF
    
    # Restrict permissions (owner read/write only)
    chmod 600 ~/.agentic.env
    
    echo ""
    echo "🔐 SECURITY: Created ~/.agentic.env with restricted permissions"
    echo "⚠️  IMPORTANT: Edit the file and add your GITHUB_TOKEN"
    echo ""
    echo "    nano ~/.agentic.env"
    echo ""
    echo "After editing, press Enter to continue..."
    read
fi

# Verify token is set
if grep -q "YOUR_TOKEN_HERE" ~/.agentic.env; then
    echo "❌ ERROR: GITHUB_TOKEN not set in ~/.agentic.env!"
    echo "Please edit: nano ~/.agentic.env"
    exit 1
fi

echo "✅ Environment file configured securely"

# Step 5: Build Docker image
echo "[5/6] Building Docker image..."
docker build -t agentic-ai-logistics:latest .

# Step 6: Run container with secure env
echo "[6/6] Starting container..."
docker run -d \
  --name agentic-app \
  --env-file ~/.agentic.env \
  -v $(pwd)/logistics_docs:/app/logistics_docs:ro \
  -v $(pwd)/chroma_db:/app/chroma_db \
  --restart unless-stopped \
  agentic-ai-logistics:latest

echo ""
echo "=========================================="
echo "✅ Deployment Complete!"
echo "=========================================="
echo ""
echo "🔐 Security Notes:"
echo "  - Token stored in: ~/.agentic.env"
echo "  - Permissions: 600 (owner read/write only)"
echo "  - NOT in Docker image or Git repository"
echo ""
echo "Check status:"
echo "  docker ps"
echo "  docker logs -f agentic-app"
echo ""
echo "Test connection:"
echo "  docker exec agentic-app python -c 'from pydantic_agents import check_ollama_status; check_ollama_status()'"
echo ""
echo "To rotate token (every 90 days):"
echo "  nano ~/.agentic.env"
echo "  docker restart agentic-app"
echo ""
