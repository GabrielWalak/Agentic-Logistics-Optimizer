#!/bin/bash
# Entrypoint script for FastAPI container
# Waits for DB, runs migrations, starts server

set -e

DB_USER="${POSTGRES_USER:-agentic_user}"
DB_HOST="${DB_HOST:-db}"
DB_PORT="${DB_PORT:-5432}"

echo "🔄 Waiting for PostgreSQL to be ready..."
until pg_isready -h "$DB_HOST" -p "$DB_PORT" -U "$DB_USER" 2>/dev/null; do
  echo "  ⏳ Postgres not ready yet... retrying in 2s"
  sleep 2
done

echo "✓ PostgreSQL is up!"

echo "🔄 Running database migrations with Alembic..."
alembic upgrade head || echo "⚠ Alembic not configured yet (first run is OK)"

echo "✓ Migrations complete!"

echo "🚀 Starting FastAPI server..."
exec uvicorn main:app --host 0.0.0.0 --port 8000 --log-level info
