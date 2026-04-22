# Multi-stage build for minimal image size
FROM python:3.11-slim as base

# Set working directory
WORKDIR /app

# Install system dependencies (minimal)
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements and install
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application files
COPY app.py .
COPY pydantic_agents.py .
COPY scenarios_examples.py .
COPY prompt_engineering.py .
COPY chroma_db_manager.py .
COPY logistics_docs/ ./logistics_docs/
COPY run.sh .
RUN chmod +x run.sh

# Create non-root user for security
RUN useradd -m -u 1000 appuser && chown -R appuser:appuser /app
USER appuser

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD python -c "from pydantic_agents import check_ollama_status; check_ollama_status()" || exit 1

# Set environment variables
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1

# Run startup script
CMD ["./run.sh"]
