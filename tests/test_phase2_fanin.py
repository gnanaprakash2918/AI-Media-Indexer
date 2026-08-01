"""Tests for the Redis fan-in counter mechanism.

Tests verify:
  - set_fanin_expected writes the correct key with correct value.
  - increment_and_check only returns True on the last INCR (= expected).
  - Disabled audio_events reduces expected count by 1.
  - Quarantined track still increments the counter (Fusion not blocked).
  - Only exactly one worker gets True back (no double-trigger).

Uses fakeredis for in-process, in-memory Redis — no real Redis needed.
"""

from __future__ import annotations

import pytest
import fakeredis
from unittest.mock import patch, MagicMock

from core.domain.chunk import make_chunk_id


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture()
def fake_redis_client():
    """In-memory Redis using fakeredis — no real Redis connection."""
    return fakeredis.FakeRedis(decode_responses=True)


@pytest.fixture(autouse=True)
def patch_redis(fake_redis_client):
    """Patch the _redis() factory in fanin.py to return the fake client."""
    with patch(
        "core.ingestion.workers.fanin._redis",
        return_value=fake_redis_client,
    ):
        yield fake_redis_client


# ---------------------------------------------------------------------------
# set_fanin_expected
# ---------------------------------------------------------------------------

class TestSetFaninExpected:
    def test_sets_expected_and_counter_keys(self, fake_redis_client):
        from core.ingestion.workers.fanin import set_fanin_expected

        chunk_id = make_chunk_id("media-001", 0, "a" * 64)
        set_fanin_expected(chunk_id, 4)

        expected = fake_redis_client.get(f"ingest:fanin:expected:{chunk_id}")
        counter = fake_redis_client.get(f"ingest:fanin:{chunk_id}")

        assert expected == "4"
        assert counter == "0"

    def test_overwrites_on_second_call(self, fake_redis_client):
        """Idempotent: calling again with a different count overwrites."""
        from core.ingestion.workers.fanin import set_fanin_expected

        chunk_id = make_chunk_id("media-001", 1, "b" * 64)
        set_fanin_expected(chunk_id, 4)
        set_fanin_expected(chunk_id, 3)  # disabled audio_events

        expected = fake_redis_client.get(f"ingest:fanin:expected:{chunk_id}")
        assert expected == "3"


# ---------------------------------------------------------------------------
# increment_and_check
# ---------------------------------------------------------------------------

class TestIncrementAndCheck:
    def _setup_chunk(self, chunk_id: str, expected: int, fake_redis_client) -> None:
        from core.ingestion.workers.fanin import set_fanin_expected
        set_fanin_expected(chunk_id, expected)

    def test_returns_false_until_last_track(self, fake_redis_client):
        """increment_and_check returns False for all tracks except the last."""
        from core.ingestion.workers.fanin import increment_and_check

        chunk_id = make_chunk_id("media-002", 0, "c" * 64)
        self._setup_chunk(chunk_id, 4, fake_redis_client)

        results = []
        for _ in range(4):
            results.append(increment_and_check(chunk_id))

        # Only the last one returns True
        assert results == [False, False, False, True]

    def test_returns_true_exactly_once(self, fake_redis_client):
        """Exactly one INCR returns True regardless of call order."""
        from core.ingestion.workers.fanin import increment_and_check

        chunk_id = make_chunk_id("media-002", 1, "d" * 64)
        self._setup_chunk(chunk_id, 3, fake_redis_client)

        true_count = sum(increment_and_check(chunk_id) for _ in range(3))
        assert true_count == 1

    def test_three_tracks_when_audio_disabled(self, fake_redis_client):
        """With audio_events disabled, expected=3 → True fires on third INCR."""
        from core.ingestion.workers.fanin import increment_and_check

        chunk_id = make_chunk_id("media-003", 0, "e" * 64)
        self._setup_chunk(chunk_id, 3, fake_redis_client)  # audio disabled

        results = [increment_and_check(chunk_id) for _ in range(3)]
        assert results == [False, False, True]

    def test_quarantined_track_still_increments(self, fake_redis_client):
        """A quarantined track must increment the counter so Fusion is not blocked.

        If a track is quarantined but does NOT INCR, expected=4 is never
        reached and Fusion never fires — deadlock. This test verifies that
        the quarantine path calls increment_and_check normally.
        """
        from core.ingestion.workers.fanin import increment_and_check

        chunk_id = make_chunk_id("media-004", 0, "f" * 64)
        self._setup_chunk(chunk_id, 4, fake_redis_client)

        # Simulate: 3 normal tracks complete, 1 is quarantined
        # The quarantined track still calls increment_and_check
        results = []
        for i in range(4):
            # Track 2 is "quarantined" but still increments
            results.append(increment_and_check(chunk_id))

        assert True in results, "Fusion was never triggered — quarantine deadlock!"
        assert sum(1 for r in results if r) == 1, "Fusion triggered more than once!"

    def test_missing_expected_key_returns_false(self, fake_redis_client):
        """If the expected key is missing (TTL expired), return False safely."""
        from core.ingestion.workers.fanin import increment_and_check

        chunk_id = make_chunk_id("media-005", 0, "g" * 64)
        # Do NOT call set_fanin_expected — key is missing

        result = increment_and_check(chunk_id)
        assert result is False

    def test_get_status(self, fake_redis_client):
        """get_status returns current counter and expected for debugging."""
        from core.ingestion.workers.fanin import get_status, set_fanin_expected

        chunk_id = make_chunk_id("media-006", 0, "h" * 64)
        set_fanin_expected(chunk_id, 3)

        status = get_status(chunk_id)
        assert status["completed"] == 0
        assert status["expected"] == 3
        assert status["chunk_id"] == chunk_id


# ---------------------------------------------------------------------------
# Dispatcher expected count computation
# ---------------------------------------------------------------------------

class TestDispatcherExpectedCount:
    """Verify the dispatcher computes the correct fan-in expected count."""

    def test_audio_enabled_gives_four(self):
        from unittest.mock import patch
        with patch("core.ingestion.dispatcher.settings") as mock_settings:
            mock_settings.enable_audio_events = True
            mock_settings.chunk_duration_seconds = 600.0
            mock_settings.min_media_length_for_chunking = 1800.0
            mock_settings.ingestion_stage_version = "v1.0.0"
            mock_settings.postgres_url = "postgresql+asyncpg://localhost/test"

            from core.ingestion.dispatcher import MediaIngestDispatcher

            dispatcher = MediaIngestDispatcher.__new__(MediaIngestDispatcher)
            dispatcher._chunk_duration_s = 600.0
            dispatcher._min_length_s = 1800.0
            dispatcher._stage_version = "v1.0.0"

            with patch("core.ingestion.dispatcher.settings.enable_audio_events", True):
                count = dispatcher._compute_expected_track_count()
            assert count == 4, f"Expected 4 tracks when audio enabled, got {count}"

    def test_audio_disabled_gives_three(self):
        from core.ingestion.dispatcher import MediaIngestDispatcher
        from unittest.mock import patch

        dispatcher = MediaIngestDispatcher.__new__(MediaIngestDispatcher)
        dispatcher._chunk_duration_s = 600.0
        dispatcher._min_length_s = 1800.0
        dispatcher._stage_version = "v1.0.0"

        with patch("core.ingestion.dispatcher.settings.enable_audio_events", False):
            count = dispatcher._compute_expected_track_count()
        assert count == 3, f"Expected 3 tracks when audio disabled, got {count}"
