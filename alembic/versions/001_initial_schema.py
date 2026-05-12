"""Initial migration: Create audit_logs and sessions tables

Revision ID: 001_initial_schema
Revises: 
Create Date: 2026-05-07 20:00:00.000000

"""
from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql

# Revision identifiers
revision = '001_initial_schema'
down_revision = None
branch_labels = None
depends_on = None


def upgrade() -> None:
    """Create initial schema"""
    
    # Sessions table
    op.create_table(
        'sessions',
        sa.Column('id', sa.String(length=36), nullable=False),
        sa.Column('session_id', sa.String(length=36), nullable=False),
        sa.Column('user_id', sa.String(), nullable=False),
        sa.Column('context_summary', sa.String(), nullable=True),
        sa.Column('session_metadata', postgresql.JSON(), nullable=False, server_default='{}'),
        sa.Column('is_active', sa.Boolean(), nullable=False, server_default='true'),
        sa.Column('created_at', sa.DateTime(timezone=True), nullable=False),
        sa.Column('last_activity', sa.DateTime(timezone=True), nullable=False),
        sa.Column('updated_at', sa.DateTime(timezone=True), nullable=False),
        sa.PrimaryKeyConstraint('id'),
        sa.UniqueConstraint('session_id'),
    )
    op.create_index('sessions_session_id_idx', 'sessions', ['session_id'])
    op.create_index('sessions_user_id_idx', 'sessions', ['user_id'])
    op.create_index('sessions_created_at_idx', 'sessions', ['created_at'])

    # Audit logs table
    op.create_table(
        'audit_logs',
        sa.Column('id', sa.String(length=36), nullable=False),
        sa.Column('session_id', sa.String(length=36), nullable=False),
        sa.Column('request_id', sa.String(length=36), nullable=False),
        sa.Column('endpoint', sa.String(), nullable=False),
        sa.Column('input_data', postgresql.JSON(), nullable=False, server_default='{}'),
        sa.Column('output_data', postgresql.JSON(), nullable=False, server_default='{}'),
        sa.Column('model_name', sa.String(), nullable=False, server_default='gpt-4o-mini'),
        sa.Column('tokens_used', sa.Integer(), nullable=False, server_default='0'),
        sa.Column('response_time_ms', sa.Integer(), nullable=False, server_default='0'),
        sa.Column('created_at', sa.DateTime(timezone=True), nullable=False),
        sa.Column('timestamp', sa.DateTime(timezone=True), nullable=False),
        sa.PrimaryKeyConstraint('id'),
    )
    op.create_index('audit_logs_session_id_idx', 'audit_logs', ['session_id'])
    op.create_index('audit_logs_request_id_idx', 'audit_logs', ['request_id'])
    op.create_index('audit_logs_timestamp_idx', 'audit_logs', ['timestamp'])


def downgrade() -> None:
    """Drop tables on downgrade"""
    op.drop_table('audit_logs')
    op.drop_table('sessions')
