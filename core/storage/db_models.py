"""SQLAlchemy table definitions for the persistent ingestion checkpoint store.

Uses core SQLAlchemy Table() objects (not ORM declarative) so this module
can be imported without triggering any database connection. All I/O lives
in the repository layer (chunk_state_repo.py).

Alembic imports this module's `metadata` object for autogenerate support.
"""

from __future__ import annotations

from sqlalchemy import (
    Column,
    Index,
    Integer,
    MetaData,
    String,
    Table,
    Text,
    TIMESTAMP,
    text,
)

metadata = MetaData()

chunk_state = Table(
    "chunk_state",
    metadata,
    # ---------------------------------------------------------------------------
    # Identity columns
    # ---------------------------------------------------------------------------
    Column(
        "chunk_id",
        String(64),
        nullable=False,
        comment="sha256(media_id:chunk_index:source_sha256) — stable across all stages",
    ),
    Column(
        "media_id",
        String(64),
        nullable=False,
        comment="Unique identifier for the parent media file",
    ),
    Column(
        "chunk_index",
        Integer,
        nullable=False,
        comment="0-based index of the time window within the media file",
    ),
    Column(
        "stage",
        String(32),
        nullable=False,
        comment="Processing stage: scene|ocr_vision|speech|audio_event|metadata|fusion",
    ),
    Column(
        "stage_version",
        String(64),
        nullable=False,
        comment="Pipeline+model version string e.g. 'v1.0.0-Qwen3-2B-whisper-large-v3'",
    ),
    # ---------------------------------------------------------------------------
    # State columns
    # ---------------------------------------------------------------------------
    Column(
        "status",
        String(16),
        nullable=False,
        server_default="pending",
        comment="pending|processing|completed|failed|quarantined",
    ),
    Column(
        "attempt_count",
        Integer,
        nullable=False,
        server_default="0",
        comment="Number of attempts made so far",
    ),
    Column(
        "max_attempts",
        Integer,
        nullable=False,
        server_default="3",
        comment="Per-stage retry limit set at enqueue time",
    ),
    Column(
        "error_message",
        Text,
        nullable=True,
        comment="Last error message — preserved across retries for observability",
    ),
    # ---------------------------------------------------------------------------
    # Timing columns (per AGENTS.md: per-stage latency metrics)
    # ---------------------------------------------------------------------------
    Column(
        "processing_started_at",
        TIMESTAMP(timezone=True),
        nullable=True,
        comment="Set when status transitions to processing — enables latency tracking",
    ),
    Column(
        "created_at",
        TIMESTAMP(timezone=True),
        nullable=False,
        server_default=text("NOW()"),
    ),
    Column(
        "updated_at",
        TIMESTAMP(timezone=True),
        nullable=False,
        server_default=text("NOW()"),
        onupdate=text("NOW()"),
    ),
    # ---------------------------------------------------------------------------
    # Primary key: (chunk_id, stage)
    # Enables idempotent upserts — the same chunk+stage can be written
    # N times and the result is always a single row.
    # ---------------------------------------------------------------------------
    *[],  # Additional constraints via Index definitions below
)

# Explicit primary key (SQLAlchemy Table API)
from sqlalchemy import PrimaryKeyConstraint  # noqa: E402

chunk_state.append_constraint(
    PrimaryKeyConstraint("chunk_id", "stage", name="pk_chunk_state")
)

# Index: find all incomplete stages for a media file (resumability query)
Index(
    "idx_chunk_state_media_stage_status",
    chunk_state.c.media_id,
    chunk_state.c.stage,
    chunk_state.c.status,
)

# Partial index: operator dead-letter review
Index(
    "idx_chunk_state_quarantined",
    chunk_state.c.status,
    postgresql_where=chunk_state.c.status == "quarantined",
)
