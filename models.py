"""
SQLModel tables for PostgreSQL persistence
Audit logs (JSONB) + Sessions metadata
"""

from datetime import datetime, timezone
from typing import Optional, Dict, Any
from uuid import uuid4

from sqlmodel import SQLModel, Field
from sqlalchemy import Column, JSON


class AuditLog(SQLModel, table=True):
    """Audit trail for all agent requests and responses"""
    __tablename__ = "audit_logs"
    
    id: str = Field(default_factory=lambda: str(uuid4()), primary_key=True)
    session_id: str = Field(index=True)
    request_id: str = Field(index=True)
    endpoint: str  # e.g., "/analyze", "/batch-analyze"
    
    # JSONB columns for flexibility
    input_data: Dict[str, Any] = Field(sa_column=Column(JSON), default_factory=dict)
    output_data: Dict[str, Any] = Field(sa_column=Column(JSON), default_factory=dict)
    
    # Metadata
    model_name: str = "unknown"
    tokens_used: int = 0
    response_time_ms: int = 0
    
    # Timestamps
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc), index=True)


class Session(SQLModel, table=True):
    """User sessions for agent conversations"""
    __tablename__ = "sessions"
    
    id: str = Field(default_factory=lambda: str(uuid4()), primary_key=True)
    session_id: str = Field(index=True, unique=True)
    user_id: str = Field(index=True)
    
    # Session metadata (JSONB for flexibility)
    context_summary: Optional[str] = None
    session_metadata: Dict[str, Any] = Field(sa_column=Column(JSON), default_factory=dict)
    
    # Status tracking
    is_active: bool = True
    
    # Timestamps
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc), index=True)
    last_activity: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    updated_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
