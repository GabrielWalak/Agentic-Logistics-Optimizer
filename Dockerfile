# Dockerfile for Agentic Logistics FastAPI
FROM python:3.11-slim

WORKDIR /app

# Install dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    bash \
    postgresql-client \
    gcc \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements
COPY requirements.txt .

# Install Python packages
RUN pip install --no-cache-dir -r requirements.txt

# Copy application files
COPY pydantic_agents.py .
COPY prompt_engineering.py .
COPY ml_predictor.py .
COPY chroma_db_manager.py .
COPY models.py .
COPY database.py .
COPY main.py .
COPY logistics_knowledge_base.py .
COPY entrypoint.sh .
COPY alembic.ini .
COPY alembic/ ./alembic/
COPY logistics_docs/ ./logistics_docs/
COPY templates/ ./templates/
COPY xgboost_model.pkl .

# Make entrypoint executable
RUN chmod +x entrypoint.sh

# Create chroma_db directory
RUN mkdir -p ./chroma_db

# Create non-root user
RUN useradd -m -u 1000 appuser && chown -R appuser:appuser /app
USER appuser

# Environment
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1
ENV ENVIRONMENT=production
ENV HOST=0.0.0.0
ENV PORT=8000

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=15s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# Expose
EXPOSE 8000

# Run entrypoint script
CMD ["/bin/bash", "entrypoint.sh"]
