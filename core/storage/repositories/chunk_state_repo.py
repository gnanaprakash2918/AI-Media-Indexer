"""Async repository for chunk_state idempotent checkpoint reads and writes.

All writes use INSERT ... ON CONFLICT (chunk_id, stage) DO UPDATE so they
are safe to call any number of times — a restarted worker cannot create
duplicate rows.

Usage:
    async with ChunkStateRepository.from_settings() as repo:
        await repo.mark_processing(chunk_id, stage, stage_version)
        # ... do work ...
        await repo.mark_completed(chunk_id, stage)

Design notes:
- No ORM Session — uses SQLAlchemy 2.x Core (connection.execute) for
  minimal overhead in a hot path.
- Engine is created lazily per repository instance and shared within a
  single async context manager.
- Tests can inject a test-only engine via ChunkStateRepository(engine=...).
"""

from __future__ import annotations

import contextlib
from datetime import datetime, timezone
from typing import Any, AsyncIterator

from sqlalchemy import select, text
from sqlalchemy.ext.asyncio import AsyncConnection, AsyncEngine, create_async_engine

from config import settings
from core.domain.chunk import ChunkStage, ChunkStatus, MAX_ATTEMPTS_BY_STAGE
from core.storage.db_models import chunk_state as chunk_state_table
from core.utils.logger import logger


def _now() -> datetime:
    return datetime.now(timezone.utc)


class ChunkStateRepository:
    """Async repository for idempotent chunk_state checkpoint writes.

    All public methods are coroutines. Acquire via the async context manager
    or pass an existing AsyncConnection for test/transaction scenarios.
    """

    def __init__(
        self,
        engine: AsyncEngine | None = None,
        *,
        connection: AsyncConnection | None = None,
    ) -> None:
        self._engine = engine
        self._connection = connection
        self._owns_engine = engine is None and connection is None

    @classmethod
    def from_settings(cls) -> "ChunkStateRepository":
        """Create a repository backed by the application postgres_url."""
        engine = create_async_engine(
            settings.postgres_url,
            pool_size=5,
            max_overflow=10,
            echo=False,
        )
        repo = cls(engine=engine)
        repo._owns_engine = True
        return repo

    @contextlib.asynccontextmanager
    async def _conn(self) -> AsyncIterator[AsyncConnection]:
        """Yield an active AsyncConnection."""
        if self._connection is not None:
            yield self._connection
        elif self._engine is not None:
            async with self._engine.begin() as conn:
                yield conn
        else:
            raise RuntimeError("ChunkStateRepository has no engine or connection.")

    async def close(self) -> None:
        """Dispose the engine if this repository owns it."""
        if self._owns_engine and self._engine is not None:
            await self._engine.dispose()

    # -----------------------------------------------------------------------
    # Write operations (all idempotent via ON CONFLICT DO UPDATE)
    # -----------------------------------------------------------------------

    async def upsert_state(
        self,
        *,
        chunk_id: str,
        media_id: str,
        chunk_index: int,
        stage: ChunkStage,
        stage_version: str,
        status: ChunkStatus = ChunkStatus.PENDING,
        max_attempts: int | None = None,
    ) -> None:
        """Insert or update a chunk_state row idempotently.

        Safe to call multiple times with the same (chunk_id, stage) — later
        calls update only the mutable columns (status, updated_at).
        The identity columns (media_id, chunk_index, stage_version) are
        preserved from the first insert.
        """
        resolved_max = (
            max_attempts
            if max_attempts is not None
            else MAX_ATTEMPTS_BY_STAGE.get(stage, 3)
        )
        now = _now()

        stmt = text(
            """
            INSERT INTO chunk_state
                (chunk_id, media_id, chunk_index, stage, stage_version,
                 status, attempt_count, max_attempts, created_at, updated_at)
            VALUES
                (:chunk_id, :media_id, :chunk_index, :stage, :stage_version,
                 :status, 0, :max_attempts, :now, :now)
            ON CONFLICT (chunk_id, stage) DO UPDATE
                SET status     = EXCLUDED.status,
                    updated_at = EXCLUDED.updated_at
            """
        )
        async with self._conn() as conn:
            await conn.execute(
                stmt,
                {
                    "chunk_id": chunk_id,
                    "media_id": media_id,
                    "chunk_index": chunk_index,
                    "stage": stage.value,
                    "stage_version": stage_version,
                    "status": status.value,
                    "max_attempts": resolved_max,
                    "now": now,
                },
            )
        logger.debug(
            f"[ChunkState] upsert chunk={chunk_id[:8]} stage={stage.value} status={status.value}"
        )

    async def mark_processing(
        self,
        chunk_id: str,
        stage: ChunkStage,
        stage_version: str,
    ) -> None:
        """Transition to PROCESSING and record processing_started_at.

        This timestamp enables per-stage latency metrics:
            latency = updated_at - processing_started_at  (on completion/failure)
        """
        now = _now()
        stmt = text(
            """
            UPDATE chunk_state
               SET status                = :status,
                   processing_started_at = :now,
                   stage_version         = :stage_version,
                   updated_at            = :now
             WHERE chunk_id = :chunk_id AND stage = :stage
            """
        )
        async with self._conn() as conn:
            await conn.execute(
                stmt,
                {
                    "status": ChunkStatus.PROCESSING.value,
                    "now": now,
                    "stage_version": stage_version,
                    "chunk_id": chunk_id,
                    "stage": stage.value,
                },
            )
        logger.debug(
            f"[ChunkState] mark_processing chunk={chunk_id[:8]} stage={stage.value}"
        )

    async def mark_completed(self, chunk_id: str, stage: ChunkStage) -> None:
        """Transition to COMPLETED."""
        await self._set_status(chunk_id, stage, ChunkStatus.COMPLETED)

    async def mark_failed(
        self,
        chunk_id: str,
        stage: ChunkStage,
        error_message: str,
    ) -> None:
        """Increment attempt_count and transition to FAILED.

        The Celery task layer decides whether to retry or quarantine based
        on attempt_count vs max_attempts (read from this row).
        """
        now = _now()
        stmt = text(
            """
            UPDATE chunk_state
               SET status        = :status,
                   attempt_count = attempt_count + 1,
                   error_message = :error,
                   updated_at    = :now
             WHERE chunk_id = :chunk_id AND stage = :stage
            """
        )
        async with self._conn() as conn:
            await conn.execute(
                stmt,
                {
                    "status": ChunkStatus.FAILED.value,
                    "error": error_message[:2000],  # cap length
                    "now": now,
                    "chunk_id": chunk_id,
                    "stage": stage.value,
                },
            )
        logger.warning(
            f"[ChunkState] mark_failed chunk={chunk_id[:8]} stage={stage.value}: {error_message[:120]}"
        )

    async def mark_quarantined(
        self,
        chunk_id: str,
        stage: ChunkStage,
        error_message: str,
    ) -> None:
        """Transition to QUARANTINED — dead stop, no more retries.

        Called when attempt_count >= max_attempts. The worker MUST NOT
        re-raise after this call (poison-pill stop). FusionAgent treats
        quarantined tracks as absent modalities and lowers confidence.
        """
        now = _now()
        stmt = text(
            """
            UPDATE chunk_state
               SET status        = :status,
                   attempt_count = attempt_count + 1,
                   error_message = :error,
                   updated_at    = :now
             WHERE chunk_id = :chunk_id AND stage = :stage
            """
        )
        async with self._conn() as conn:
            await conn.execute(
                stmt,
                {
                    "status": ChunkStatus.QUARANTINED.value,
                    "error": error_message[:2000],
                    "now": now,
                    "chunk_id": chunk_id,
                    "stage": stage.value,
                },
            )
        logger.error(
            f"[ChunkState] QUARANTINED chunk={chunk_id[:8]} stage={stage.value}: {error_message[:120]}"
        )

    # -----------------------------------------------------------------------
    # Read operations
    # -----------------------------------------------------------------------

    async def get_row(
        self, chunk_id: str, stage: ChunkStage
    ) -> dict[str, Any] | None:
        """Fetch the current state row for one (chunk_id, stage) pair."""
        stmt = select(chunk_state_table).where(
            chunk_state_table.c.chunk_id == chunk_id,
            chunk_state_table.c.stage == stage.value,
        )
        async with self._conn() as conn:
            result = await conn.execute(stmt)
            row = result.mappings().first()
            return dict(row) if row else None

    async def get_resumable_chunks(
        self, media_id: str
    ) -> list[dict[str, Any]]:
        """Return all pending or failed (not quarantined, not completed) rows.

        Used by the dispatcher at startup to resume a partially-processed
        media file without re-queueing completed chunks.
        """
        stmt = select(chunk_state_table).where(
            chunk_state_table.c.media_id == media_id,
            chunk_state_table.c.status.in_(
                [ChunkStatus.PENDING.value, ChunkStatus.FAILED.value]
            ),
        )
        async with self._conn() as conn:
            result = await conn.execute(stmt)
            return [dict(row) for row in result.mappings()]

    async def is_completed(self, chunk_id: str, stage: ChunkStage) -> bool:
        """Return True if this chunk+stage has already completed successfully."""
        row = await self.get_row(chunk_id, stage)
        return row is not None and row["status"] == ChunkStatus.COMPLETED.value

    async def get_attempt_count(self, chunk_id: str, stage: ChunkStage) -> int:
        """Return current attempt_count for a chunk+stage row (0 if not found)."""
        row = await self.get_row(chunk_id, stage)
        return row["attempt_count"] if row else 0

    async def should_quarantine(self, chunk_id: str, stage: ChunkStage) -> bool:
        """Return True if attempt_count has reached max_attempts."""
        row = await self.get_row(chunk_id, stage)
        if not row:
            return False
        return row["attempt_count"] >= row["max_attempts"]

    # -----------------------------------------------------------------------
    # Internal helpers
    # -----------------------------------------------------------------------


    async def get_progress_for_media(self, media_id: str) -> dict[str, Any]:
        """Calculates progress based on chunk states for a given media_id."""
        async for conn in self._conn():
            stmt = select(
                chunk_state_table.c.status,
                text("COUNT(*)")
            ).where(
                chunk_state_table.c.media_id == media_id
            ).group_by(
                chunk_state_table.c.status
            )
            result = await conn.execute(stmt)
            counts = {row[0]: row[1] for row in result.fetchall()}
            
            total = sum(counts.values())
            if total == 0:
                return {"status": "pending", "progress": 0.0, "total_chunks": 0, "completed": 0}
            
            completed = counts.get("completed", 0)
            failed = counts.get("failed", 0)
            processing = counts.get("processing", 0)
            
            progress = (completed / total) * 100.0
            
            status = "completed" if completed == total else ("failed" if failed > 0 else ("running" if processing > 0 else "pending"))
            
            return {
                "status": status,
                "progress": round(progress, 2),
                "total_chunks": total,
                "completed_chunks": completed,
                "failed_chunks": failed,
                "processing_chunks": processing
            }
        return {"status": "pending", "progress": 0.0}

    async def get_all_active_media_ids(self) -> list[str]:
        """Returns all media_ids that have chunks."""
        async for conn in self._conn():
            stmt = select(chunk_state_table.c.media_id).distinct()
            result = await conn.execute(stmt)
            return [row[0] for row in result.fetchall()]
        return []

    async def _set_status(
        self,
        chunk_id: str,
        stage: ChunkStage,
        status: ChunkStatus,
    ) -> None:
        now = _now()
        stmt = text(
            """
            UPDATE chunk_state
               SET status = :status, updated_at = :now
             WHERE chunk_id = :chunk_id AND stage = :stage
            """
        )
        async with self._conn() as conn:
            await conn.execute(
                stmt,
                {
                    "status": status.value,
                    "now": now,
                    "chunk_id": chunk_id,
                    "stage": stage.value,
                },
            )
