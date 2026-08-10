"""
Database configuration and connection management
Async PostgreSQL with asyncpg driver
"""

import os
from typing import AsyncGenerator

from sqlalchemy import text
from sqlalchemy.ext.asyncio import (
    AsyncSession,
    async_sessionmaker,
    create_async_engine,
)
from sqlmodel import SQLModel


# Build async database URL
DATABASE_URL = os.getenv(
    "DATABASE_URL",
    "postgresql+asyncpg://agentic_user:agentic_password@db:5432/logistics_app"
)

# Create async engine
engine = create_async_engine(
    DATABASE_URL,
    echo=False,  # Set to True for SQL debugging
    future=True,
    pool_size=20,
    max_overflow=40,
    pool_pre_ping=True,  # Verify connections are alive before using
)

# Session factory for dependency injection
async_session_factory = async_sessionmaker(
    bind=engine,
    expire_on_commit=False,
    autoflush=False,
)


async def init_db() -> None:
    """Initialize database tables"""
    async with engine.begin() as conn:
        await conn.run_sync(SQLModel.metadata.create_all)


async def check_database_health() -> bool:
    """Execute a lightweight query to verify the current database connection."""
    try:
        async with engine.connect() as connection:
            await connection.execute(text("SELECT 1"))
        return True
    except Exception:
        return False


async def get_session() -> AsyncGenerator[AsyncSession, None]:
    """Dependency for FastAPI to get async session"""
    async with async_session_factory() as session:
        yield session
