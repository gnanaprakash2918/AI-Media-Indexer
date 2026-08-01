"""001 — Add chunk_state table.

This table provides idempotent checkpoints for the parallel ingestion DAG.
Each row represents one processing stage of one time window (chunk) of one
media file.

The PRIMARY KEY is (chunk_id, stage): upserts on this key are safe to
re-run any number of times — a crashed worker that restarts never creates
duplicate rows.

chunk_id = sha256(media_id:chunk_index:source_sha256)
         → stable across all stages and version bumps
         → stage_version is a column on the row, not part of the ID

Revision: 001
"""

from __future__ import annotations

import sqlalchemy as sa
from alembic import op

# Alembic revision identifiers
revision: str = "001"
down_revision: str | None = None  # first migration
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.create_table(
        "chunk_state",
        # --- Identity ---
        sa.Column(
            "chunk_id",
            sa.String(64),
            nullable=False,
            comment="sha256(media_id:chunk_index:source_sha256) — stable across all stages",
        ),
        sa.Column("media_id", sa.String(64), nullable=False),
        sa.Column("chunk_index", sa.Integer, nullable=False),
        sa.Column(
            "stage",
            sa.String(32),
            nullable=False,
            comment="scene|ocr_vision|speech|audio_event|metadata|fusion",
        ),
        sa.Column(
            "stage_version",
            sa.String(64),
            nullable=False,
            comment="Pipeline+model version e.g. 'v1.0.0-Qwen3-2B'",
        ),
        # --- State ---
        sa.Column(
            "status",
            sa.String(16),
            nullable=False,
            server_default="pending",
            comment="pending|processing|completed|failed|quarantined",
        ),
        sa.Column(
            "attempt_count",
            sa.Integer,
            nullable=False,
            server_default="0",
        ),
        sa.Column(
            "max_attempts",
            sa.Integer,
            nullable=False,
            server_default="3",
            comment="Per-stage retry limit — set at enqueue time from MAX_ATTEMPTS_BY_STAGE",
        ),
        sa.Column("error_message", sa.Text, nullable=True),
        # --- Timing (per-stage latency metrics per AGENTS.md) ---
        sa.Column(
            "processing_started_at",
            sa.TIMESTAMP(timezone=True),
            nullable=True,
            comment="Set on status→processing; (updated_at - processing_started_at) = stage duration",
        ),
        sa.Column(
            "created_at",
            sa.TIMESTAMP(timezone=True),
            nullable=False,
            server_default=sa.text("NOW()"),
        ),
        sa.Column(
            "updated_at",
            sa.TIMESTAMP(timezone=True),
            nullable=False,
            server_default=sa.text("NOW()"),
        ),
        # PRIMARY KEY: idempotent upsert key
        sa.PrimaryKeyConstraint("chunk_id", "stage", name="pk_chunk_state"),
    )

    # Index: resumability query — find incomplete stages for a media_id
    op.create_index(
        "idx_chunk_state_media_stage_status",
        "chunk_state",
        ["media_id", "stage", "status"],
    )

    # Partial index: operator dead-letter review — only quarantined rows
    op.create_index(
        "idx_chunk_state_quarantined",
        "chunk_state",
        ["status"],
        postgresql_where=sa.text("status = 'quarantined'"),
    )


def downgrade() -> None:
    op.drop_index("idx_chunk_state_quarantined", table_name="chunk_state")
    op.drop_index("idx_chunk_state_media_stage_status", table_name="chunk_state")
    op.drop_table("chunk_state")
