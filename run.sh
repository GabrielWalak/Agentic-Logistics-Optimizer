#!/bin/bash
# Docker entry point for AgenticAI Logistics System

set -e

echo "=========================================="
echo "AgenticAI Logistics - Docker Startup"
echo "=========================================="

# Check provider API key
if [ -z "$LLM_API_KEY" ]; then
    echo "❌ ERROR: LLM_API_KEY not set!"
    echo "Set it in .env or pass with -e LLM_API_KEY=xxx"
    exit 1
fi

echo "✓ LLM_API_KEY detected"
echo "✓ LLM API: $LLM_BASE_URL"
echo "✓ Model: $LLM_MODEL"

# Verify connection
echo ""
echo "Checking LLM API configuration..."
python3 -c "
from pydantic_agents import check_ollama_status
if check_ollama_status():
    print('✓ LLM API configured successfully!')
else:
    print('❌ LLM API configuration is incomplete')
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
