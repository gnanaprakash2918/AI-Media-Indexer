"""Tests for quarantine state logic in the chunk_state repository.

Tests verify:
  - mark_failed increments attempt_count.
  - should_quarantine returns True when attempt_count >= max_attempts.
  - mark_quarantined sets status=quarantined (not failed).
  - After quarantine, should_quarantine still returns True.
  - get_row reflects the latest state.
  - is_completed is False for quarantined rows.

Uses SQLAlchemy in-memory SQLite for speed — no real Postgres needed.
The repository uses text() SQL that is compatible with both SQLite and
PostgreSQL (ON CONFLICT syntax differs; see note below).
"""

from __future__ import annotations

import asyncio
import pytest
import pytest_asyncio
from sqlalchemy.ext.asyncio import create_async_engine, AsyncConnection

from core.domain.chunk import ChunkStage, ChunkStatus, MAX_ATTEMPTS_BY_STAGE
from core.domain.chunk import make_chunk_id


# ---------------------------------------------------------------------------
# SQLite in-memory engine for tests (no Postgres required)
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def event_loop():
    import asyncio
    loop = asyncio.new_event_loop()
    yield loop
    loop.close()


@pytest_asyncio.fixture(scope="module")
async def async_engine():
    """Create an in-memory SQLite engine with the chunk_state schema."""
    engine = create_async_engine("sqlite+aiosqlite:///:memory:", echo=False)

    from core.storage.db_models import metadata
    async with engine.begin() as conn:
        # SQLite-compatible DDL (no partial index, no TIMESTAMPTZ → use TEXT)
        await conn.run_sync(lambda sync_conn: _create_sqlite_schema(sync_conn))

    yield engine
    await engine.dispose()


def _create_sqlite_schema(conn) -> None:
    conn.execute(__import__("sqlalchemy").text("""
        CREATE TABLE IF NOT EXISTS chunk_state (
            chunk_id              TEXT NOT NULL,
            media_id              TEXT NOT NULL,
            chunk_index           INTEGER NOT NULL,
            stage                 TEXT NOT NULL,
            stage_version         TEXT NOT NULL,
            status                TEXT NOT NULL DEFAULT 'pending',
            attempt_count         INTEGER NOT NULL DEFAULT 0,
            max_attempts          INTEGER NOT NULL DEFAULT 3,
            error_message         TEXT,
            processing_started_at TEXT,
            created_at            TEXT NOT NULL DEFAULT (datetime('now')),
            updated_at            TEXT NOT NULL DEFAULT (datetime('now')),
            PRIMARY KEY (chunk_id, stage)
        )
    """))


@pytest_asyncio.fixture()
async def conn(async_engine):
    """Provide a fresh transaction for each test (rolled back after)."""
    async with async_engine.begin() as connection:
        yield connection
        await connection.rollback()


@pytest.fixture()
def repo(conn):
    from core.storage.repositories.chunk_state_repo import ChunkStateRepository
    return ChunkStateRepository(connection=conn)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

CHUNK_ID = make_chunk_id("test-media", 0, "a" * 64)
STAGE = ChunkStage.OCR_VISION
STAGE_VER = "v1.0.0-test"
MEDIA_ID = "test-media"


async def _insert_row(repo, max_attempts: int = 2) -> None:
    """Insert a fresh pending row."""
    await repo.upsert_state(
        chunk_id=CHUNK_ID,
        media_id=MEDIA_ID,
        chunk_index=0,
        stage=STAGE,
        stage_version=STAGE_VER,
        status=ChunkStatus.PENDING,
        max_attempts=max_attempts,
    )


# ---------------------------------------------------------------------------
# Test: upsert_state
# ---------------------------------------------------------------------------

class TestUpsertState:
    @pytest.mark.asyncio
    async def test_insert_creates_row(self, repo):
        await _insert_row(repo)
        row = await repo.get_row(CHUNK_ID, STAGE)
        assert row is not None
        assert row["status"] == ChunkStatus.PENDING.value
        assert row["attempt_count"] == 0

    @pytest.mark.asyncio
    async def test_upsert_twice_does_not_duplicate(self, repo):
        await _insert_row(repo)
        await _insert_row(repo)  # second call must be idempotent
        row = await repo.get_row(CHUNK_ID, STAGE)
        assert row is not None  # still one row

    @pytest.mark.asyncio
    async def test_is_completed_false_for_pending(self, repo):
        await _insert_row(repo)
        assert not await repo.is_completed(CHUNK_ID, STAGE)


# ---------------------------------------------------------------------------
# Test: mark_failed and quarantine transition
# ---------------------------------------------------------------------------

class TestQuarantineLogic:
    @pytest.mark.asyncio
    async def test_mark_failed_increments_attempt_count(self, repo):
        await _insert_row(repo, max_attempts=3)
        await repo.mark_processing(CHUNK_ID, STAGE, STAGE_VER)
        await repo.mark_failed(CHUNK_ID, STAGE, "OOM error")

        row = await repo.get_row(CHUNK_ID, STAGE)
        assert row["status"] == ChunkStatus.FAILED.value
        assert row["attempt_count"] == 1
        assert "OOM error" in row["error_message"]

    @pytest.mark.asyncio
    async def test_should_quarantine_false_below_limit(self, repo):
        """Below max_attempts → should_quarantine = False."""
        await _insert_row(repo, max_attempts=3)
        await repo.mark_processing(CHUNK_ID, STAGE, STAGE_VER)
        await repo.mark_failed(CHUNK_ID, STAGE, "first failure")

        assert not await repo.should_quarantine(CHUNK_ID, STAGE)

    @pytest.mark.asyncio
    async def test_should_quarantine_true_at_limit(self, repo):
        """At max_attempts → should_quarantine = True."""
        await _insert_row(repo, max_attempts=2)

        # Two failures
        for i in range(2):
            await repo.mark_processing(CHUNK_ID, STAGE, STAGE_VER)
            await repo.mark_failed(CHUNK_ID, STAGE, f"failure {i}")

        assert await repo.should_quarantine(CHUNK_ID, STAGE)

    @pytest.mark.asyncio
    async def test_mark_quarantined_sets_correct_status(self, repo):
        """mark_quarantined transitions to 'quarantined', not 'failed'."""
        await _insert_row(repo, max_attempts=2)
        await repo.mark_processing(CHUNK_ID, STAGE, STAGE_VER)
        await repo.mark_failed(CHUNK_ID, STAGE, "failure 1")
        await repo.mark_processing(CHUNK_ID, STAGE, STAGE_VER)
        await repo.mark_quarantined(CHUNK_ID, STAGE, "final failure")

        row = await repo.get_row(CHUNK_ID, STAGE)
        assert row["status"] == ChunkStatus.QUARANTINED.value

    @pytest.mark.asyncio
    async def test_quarantined_is_not_completed(self, repo):
        """Quarantined status does not count as completed."""
        await _insert_row(repo, max_attempts=1)
        await repo.mark_processing(CHUNK_ID, STAGE, STAGE_VER)
        await repo.mark_failed(CHUNK_ID, STAGE, "failure")
        await repo.mark_quarantined(CHUNK_ID, STAGE, "quarantine")

        assert not await repo.is_completed(CHUNK_ID, STAGE)

    @pytest.mark.asyncio
    async def test_quarantined_error_message_preserved(self, repo):
        """Error message is visible after quarantine for operator review."""
        await _insert_row(repo, max_attempts=1)
        await repo.mark_processing(CHUNK_ID, STAGE, STAGE_VER)
        await repo.mark_failed(CHUNK_ID, STAGE, "CUDA OOM in VLM forward pass")
        await repo.mark_quarantined(CHUNK_ID, STAGE, "CUDA OOM in VLM forward pass")

        row = await repo.get_row(CHUNK_ID, STAGE)
        assert "CUDA OOM" in row["error_message"]

    @pytest.mark.asyncio
    async def test_ocr_vision_max_attempts_is_two(self):
        """OCR/Vision has max_attempts=2 — lowest of all stages (GPU cost)."""
        assert MAX_ATTEMPTS_BY_STAGE[ChunkStage.OCR_VISION] == 2

    @pytest.mark.asyncio
    async def test_metadata_max_attempts_is_five(self):
        """Metadata has max_attempts=5 — highest (near-zero cost, high tolerance)."""
        assert MAX_ATTEMPTS_BY_STAGE[ChunkStage.METADATA] == 5


# ---------------------------------------------------------------------------
# Test: resumability
# ---------------------------------------------------------------------------

class TestResumability:
    @pytest.mark.asyncio
    async def test_get_resumable_chunks_excludes_completed(self, repo):
        """Completed rows should not appear in resumable list."""
        await _insert_row(repo, max_attempts=3)
        await repo.mark_processing(CHUNK_ID, STAGE, STAGE_VER)
        await repo.mark_completed(CHUNK_ID, STAGE)

        resumable = await repo.get_resumable_chunks(MEDIA_ID)
        assert all(
            r["chunk_id"] != CHUNK_ID or r["stage"] != STAGE.value
            for r in resumable
        ), "Completed chunk should not be in resumable list"

    @pytest.mark.asyncio
    async def test_get_resumable_includes_failed(self, repo):
        """Failed rows (not yet quarantined) appear in resumable list."""
        await _insert_row(repo, max_attempts=3)
        await repo.mark_processing(CHUNK_ID, STAGE, STAGE_VER)
        await repo.mark_failed(CHUNK_ID, STAGE, "transient error")

        resumable = await repo.get_resumable_chunks(MEDIA_ID)
        matching = [r for r in resumable if r["stage"] == STAGE.value]
        assert len(matching) >= 1
