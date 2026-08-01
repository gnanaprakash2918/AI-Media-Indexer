"""Tests for FusionAgent temporal alignment merge logic.

Tests verify:
  - Subtitle cue wins over Whisper segment when overlap >= 80%.
  - Whisper segment is kept when subtitle overlap < 80%.
  - Non-overlapping subtitle cues are added as standalone segments.
  - Confidence is lower when modalities are absent.
  - Quarantine penalty is applied when any stage is absent.
  - Output is sorted by start_ms.
  - Subtitle source_type is preserved in merged output.
"""

from __future__ import annotations

import pytest

from core.ingestion.workers.fusion_worker import (
    _SUBTITLE_OVERLAP_THRESHOLD,
    _apply_subtitle_wins,
    _compute_confidence,
)


# ---------------------------------------------------------------------------
# Subtitle-wins merge rule
# ---------------------------------------------------------------------------

class TestSubtitleWins:
    def _seg(self, start_ms: int, end_ms: int, text: str, source: str = "whisper") -> dict:
        return {"start_ms": start_ms, "end_ms": end_ms, "text": text, "source_type": source, "speaker": ""}

    def _cue(self, start_ms: int, end_ms: int, text: str) -> dict:
        return {"start_ms": start_ms, "end_ms": end_ms, "text": text}

    def test_subtitle_replaces_whisper_when_overlap_sufficient(self):
        """Subtitle cue overlapping Whisper by >=80% → subtitle text wins."""
        speech = [self._seg(0, 3000, "whisper text")]
        subtitles = [self._cue(0, 3000, "subtitle text")]

        merged = _apply_subtitle_wins(speech, subtitles)

        assert len(merged) == 1
        assert merged[0]["text"] == "subtitle text"
        assert merged[0]["source_type"] == "subtitle"

    def test_whisper_kept_when_overlap_insufficient(self):
        """Subtitle cue overlapping Whisper by <80% → Whisper text kept."""
        # Subtitle: 0–1000ms, Whisper: 0–3000ms → overlap = 1000/1000 = 100% FROM CUE perspective
        # But let's make the overlap fraction < 0.8 for the cue duration
        # Cue is 3000ms long, overlap with segment is only 1000ms → 1000/3000 = 33%
        speech = [self._seg(2000, 5000, "whisper text")]
        subtitles = [self._cue(0, 3000, "subtitle text")]  # overlap 2000-3000 = 1000ms, cue is 3000ms → 33%

        merged = _apply_subtitle_wins(speech, subtitles)

        # The whisper segment stays as whisper
        whisper_segs = [s for s in merged if s["source_type"] == "whisper"]
        assert len(whisper_segs) == 1
        assert whisper_segs[0]["text"] == "whisper text"

    def test_non_overlapping_subtitle_added_as_standalone(self):
        """Subtitle cue with no matching Whisper segment → added as new segment."""
        speech = [self._seg(0, 1000, "word one")]
        subtitles = [self._cue(5000, 7000, "separate subtitle")]

        merged = _apply_subtitle_wins(speech, subtitles)

        texts = [s["text"] for s in merged]
        assert "word one" in texts
        assert "separate subtitle" in texts

    def test_output_sorted_by_start_ms(self):
        """Merged output is always sorted by start_ms."""
        speech = [self._seg(5000, 7000, "b"), self._seg(0, 2000, "a")]
        subtitles = [self._cue(3000, 4000, "mid")]

        merged = _apply_subtitle_wins(speech, subtitles)

        starts = [s["start_ms"] for s in merged]
        assert starts == sorted(starts), "Output must be sorted by start_ms"

    def test_multiple_subtitle_cues_merged_correctly(self):
        """Multiple subtitle cues, each matching one Whisper segment."""
        speech = [
            self._seg(0, 2000, "first whisper"),
            self._seg(3000, 5000, "second whisper"),
        ]
        subtitles = [
            self._cue(0, 2000, "first subtitle"),
            self._cue(3000, 5000, "second subtitle"),
        ]

        merged = _apply_subtitle_wins(speech, subtitles)

        texts = {s["text"] for s in merged}
        assert "first subtitle" in texts
        assert "second subtitle" in texts
        assert "first whisper" not in texts
        assert "second whisper" not in texts

    def test_no_subtitles_returns_speech_unchanged(self):
        """With no subtitle cues, speech segments are returned as-is."""
        speech = [self._seg(0, 1000, "only whisper")]
        merged = _apply_subtitle_wins(speech, [])

        assert len(merged) == 1
        assert merged[0]["text"] == "only whisper"
        assert merged[0]["source_type"] == "whisper"

    def test_no_speech_all_subtitles(self):
        """With no Whisper segments, all subtitle cues become standalone segments."""
        subtitles = [
            self._cue(0, 2000, "first"),
            self._cue(3000, 5000, "second"),
        ]
        merged = _apply_subtitle_wins([], subtitles)

        assert len(merged) == 2
        assert all(s["source_type"] == "subtitle" for s in merged)

    def test_overlap_threshold_value(self):
        """Overlap threshold is 80% — changing it is a breaking change."""
        assert _SUBTITLE_OVERLAP_THRESHOLD == 0.8


# ---------------------------------------------------------------------------
# Confidence computation
# ---------------------------------------------------------------------------

class TestComputeConfidence:
    def test_all_modalities_present_gives_max_confidence(self):
        """When all 4 modalities are present, confidence = 1.0 (no penalty)."""
        conf = _compute_confidence(
            has_speech=True,
            has_vision=True,
            has_ocr=True,
            has_audio=True,
            absent_stages=[],
        )
        assert conf == 1.0

    def test_no_modalities_gives_zero(self):
        """No modalities present → confidence = 0.0."""
        conf = _compute_confidence(
            has_speech=False,
            has_vision=False,
            has_ocr=False,
            has_audio=False,
            absent_stages=[],
        )
        assert conf == 0.0

    def test_quarantine_penalty_applied(self):
        """Quarantined stage lowers confidence by quarantine penalty."""
        from core.domain.chunk import ChunkStage

        conf_clean = _compute_confidence(
            has_speech=True,
            has_vision=True,
            has_ocr=False,
            has_audio=False,
            absent_stages=[],
        )
        conf_quarantine = _compute_confidence(
            has_speech=True,
            has_vision=True,
            has_ocr=False,
            has_audio=False,
            absent_stages=[ChunkStage.SCENE],  # quarantined
        )
        assert conf_quarantine < conf_clean, (
            "Confidence must be lower when a stage is quarantined"
        )

    def test_speech_only_is_positive(self):
        """Speech-only event still has positive confidence."""
        conf = _compute_confidence(
            has_speech=True,
            has_vision=False,
            has_ocr=False,
            has_audio=False,
            absent_stages=[],
        )
        assert 0.0 < conf < 1.0

    def test_confidence_in_range(self):
        """Confidence is always in [0.0, 1.0]."""
        from core.domain.chunk import ChunkStage

        for has_s, has_v, has_o, has_a, absent in [
            (True, True, True, True, []),
            (True, False, False, False, []),
            (False, False, False, False, [ChunkStage.SCENE]),
            (True, True, False, False, [ChunkStage.OCR_VISION]),
        ]:
            conf = _compute_confidence(
                has_speech=has_s,
                has_vision=has_v,
                has_ocr=has_o,
                has_audio=has_a,
                absent_stages=absent,
            )
            assert 0.0 <= conf <= 1.0, f"Out of range: {conf}"
