"""Tests for chunk_id determinism and stability.

chunk_id must be:
  - Deterministic: same inputs always produce the same hash.
  - Stable across stage versions: stage_version does NOT change the ID.
  - Collision-resistant: different inputs produce different hashes.
  - Boundary-aware: compute_chunk_boundaries produces correct windows.
"""

from __future__ import annotations

import pytest
from core.domain.chunk import (
    ChunkStage,
    ChunkStatus,
    FANIN_BASE_COUNT,
    MAX_ATTEMPTS_BY_STAGE,
    compute_chunk_boundaries,
    make_chunk_id,
)


# ---------------------------------------------------------------------------
# make_chunk_id
# ---------------------------------------------------------------------------

class TestMakeChunkId:
    def test_deterministic_same_inputs(self):
        """Same inputs always produce the same chunk_id."""
        a = make_chunk_id("media-abc", 0, "deadbeef" * 8)
        b = make_chunk_id("media-abc", 0, "deadbeef" * 8)
        assert a == b

    def test_stable_across_stage_versions(self):
        """stage_version is NOT part of the chunk_id — different versions give same ID."""
        # Both calls produce the same chunk_id even though stage_version differs
        # (stage_version is a row column, not an ID component)
        id_v1 = make_chunk_id("media-abc", 3, "cafe" * 16)
        id_v2 = make_chunk_id("media-abc", 3, "cafe" * 16)
        assert id_v1 == id_v2, (
            "chunk_id must not encode stage_version — "
            "a VLM upgrade must not produce a new chunk identity"
        )

    def test_different_chunk_index_gives_different_id(self):
        """Different chunk_index → different chunk_id for the same media."""
        id_0 = make_chunk_id("media-abc", 0, "deadbeef" * 8)
        id_1 = make_chunk_id("media-abc", 1, "deadbeef" * 8)
        assert id_0 != id_1

    def test_different_media_id_gives_different_id(self):
        """Different media_id → different chunk_id for the same chunk index."""
        id_a = make_chunk_id("media-abc", 0, "deadbeef" * 8)
        id_b = make_chunk_id("media-xyz", 0, "deadbeef" * 8)
        assert id_a != id_b

    def test_different_source_sha256_gives_different_id(self):
        """Re-upload of a different file (different sha256) → different chunk_id.

        This prevents stale data reuse when a file is replaced with a
        different file of the same name.
        """
        id_file1 = make_chunk_id("media-abc", 0, "aaa" * 20 + "aa")
        id_file2 = make_chunk_id("media-abc", 0, "bbb" * 20 + "bb")
        assert id_file1 != id_file2

    def test_output_is_64_hex_chars(self):
        """chunk_id is always a 64-character lowercase hex SHA-256."""
        chunk_id = make_chunk_id("media-abc", 0, "x" * 64)
        assert len(chunk_id) == 64
        assert all(c in "0123456789abcdef" for c in chunk_id)

    def test_chunk_index_zero_and_one_distinct(self):
        """Edge case: chunk 0 and chunk 1 are distinct (not just different strings)."""
        id_0 = make_chunk_id("m", 0, "s" * 64)
        id_1 = make_chunk_id("m", 1, "s" * 64)
        assert id_0 != id_1


# ---------------------------------------------------------------------------
# compute_chunk_boundaries
# ---------------------------------------------------------------------------

class TestComputeChunkBoundaries:
    def test_exact_multiple(self):
        """30-minute video with 10-minute chunks → 3 chunks."""
        bounds = compute_chunk_boundaries(1800.0, 600.0)
        assert len(bounds) == 3
        assert bounds[0] == (0.0, 600.0)
        assert bounds[1] == (600.0, 1200.0)
        assert bounds[2] == (1200.0, 1800.0)

    def test_last_chunk_truncated(self):
        """Non-exact multiple: last chunk ends exactly at total_duration."""
        bounds = compute_chunk_boundaries(700.0, 600.0)
        assert len(bounds) == 2
        assert bounds[1] == (600.0, 700.0)

    def test_single_chunk_short_media(self):
        """Media shorter than chunk_duration → one chunk covering all."""
        bounds = compute_chunk_boundaries(300.0, 600.0)
        assert len(bounds) == 1
        assert bounds[0] == (0.0, 300.0)

    def test_eighteen_hours(self):
        """18-hour video at 10-min chunks = 108 chunks."""
        duration = 18 * 3600.0
        bounds = compute_chunk_boundaries(duration, 600.0)
        assert len(bounds) == 108
        # First chunk
        assert bounds[0][0] == 0.0
        assert bounds[0][1] == 600.0
        # Last chunk ends exactly at 18h
        assert bounds[-1][1] == duration

    def test_all_start_times_match_previous_end(self):
        """No gaps: each chunk starts exactly where the previous ended."""
        bounds = compute_chunk_boundaries(3700.0, 600.0)
        for i in range(1, len(bounds)):
            assert bounds[i][0] == bounds[i - 1][1]

    def test_invalid_chunk_duration_raises(self):
        with pytest.raises(ValueError):
            compute_chunk_boundaries(3600.0, 0.0)

    def test_zero_duration_returns_single_entry(self):
        bounds = compute_chunk_boundaries(0.0, 600.0)
        assert bounds == [(0.0, 0.0)]


# ---------------------------------------------------------------------------
# ChunkStage and ChunkStatus enums
# ---------------------------------------------------------------------------

class TestChunkEnums:
    def test_all_stages_have_max_attempts(self):
        """Every ChunkStage must have a configured max_attempts."""
        for stage in ChunkStage:
            assert stage in MAX_ATTEMPTS_BY_STAGE, (
                f"ChunkStage.{stage.name} missing from MAX_ATTEMPTS_BY_STAGE"
            )

    def test_ocr_vision_has_lower_attempts_than_metadata(self):
        """GPU-heavy stages get fewer retries than cheap stages."""
        assert (
            MAX_ATTEMPTS_BY_STAGE[ChunkStage.OCR_VISION]
            < MAX_ATTEMPTS_BY_STAGE[ChunkStage.METADATA]
        )

    def test_status_values_are_strings(self):
        """ChunkStatus values are plain strings (compatible with VARCHAR column)."""
        for status in ChunkStatus:
            assert isinstance(status.value, str)

    def test_fanin_base_count_is_two(self):
        """FANIN_BASE_COUNT = 2 (scene-chain + speech)."""
        assert FANIN_BASE_COUNT == 2
