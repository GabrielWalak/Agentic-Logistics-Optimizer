#!/bin/bash
# Docker entry point for AgenticAI Logistics System

set -e

echo "=========================================="
echo "AgenticAI Logistics - Docker Startup"
echo "=========================================="

# Check GitHub token
if [ -z "$GITHUB_TOKEN" ]; then
    echo "❌ ERROR: GITHUB_TOKEN not set!"
    echo "Set it in .env or pass with -e GITHUB_TOKEN=xxx"
    exit 1
fi

echo "✓ GITHUB_TOKEN detected"
echo "✓ GitHub Models API: $GITHUB_MODELS_BASE_URL"
echo "✓ Model: $GITHUB_MODEL"

# Verify connection
echo ""
echo "Checking GitHub Models API connection..."
python3 -c "
from pydantic_agents import check_ollama_status
if check_ollama_status():
    print('✓ GitHub Models API connected successfully!')
else:
    print('❌ Failed to connect to GitHub Models API')
    exit(1)
" || exit 1

echo ""
echo "=========================================="
echo "System Status: READY"
echo "=========================================="
echo ""
echo "Available commands:"
echo "  python app.py          - Interactive menu"
echo "  python test_grader.py  - Test grading framework"
echo ""
echo "For interactive shell:"
echo "  docker exec -it agentic-app python app.py"
echo ""
echo "Container is running. Press Ctrl+C to stop."
echo ""

# Run app in interactive mode if terminal available
if [ -t 0 ]; then
    python app.py
else
    # If not interactive (background), keep container running
    echo "Running in background mode..."
    tail -f /dev/null
fi
