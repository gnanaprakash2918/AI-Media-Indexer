"""Chunk domain types and identity functions.

A chunk is a fixed-duration time window of a media file used as the unit of
parallel processing. chunk_id is stable: it does NOT include stage_version or
model version so that all stages of the same time window share one identity.

stage_version is a per-row column in chunk_state, not part of the ID.
"""

from __future__ import annotations

import hashlib
from enum import Enum


class ChunkStage(str, Enum):
    """Extraction stages in the ingestion DAG.

    Ordered by execution: scene runs first (produces keyframes needed by
    ocr_vision). speech/audio_event/metadata run in parallel with scene.
    fusion runs last, after all tracks complete.

    Adding a new stage: add it here + update MAX_ATTEMPTS_BY_STAGE +
    update dispatcher.py fan-in expected count. No DB migration needed
    (stage column is VARCHAR with no enum constraint).
    """

    SCENE = "scene"          # SceneAgent: shot boundaries + keyframes
    OCR_VISION = "ocr_vision"  # OCRAgent + VisionAgent (depends on SCENE)
    SPEECH = "speech"        # SpeechAgent + VoiceAgent (parallel with scene)
    AUDIO_EVENT = "audio_event"  # AudioEventAgent / CLAP (optional)
    METADATA = "metadata"    # MetadataAgent + SubtitleParser
    FUSION = "fusion"        # FusionAgent: temporal alignment merge


class ChunkStatus(str, Enum):
    """Lifecycle states of a chunk_state row.

    Transitions:
        pending -> processing -> completed
        processing -> failed   (attempt_count < max_attempts → Celery retries)
        processing -> quarantined  (attempt_count >= max_attempts → dead stop)

    Quarantined chunks are treated as "absent modality" by FusionAgent:
    it proceeds with remaining tracks and lowers event confidence.
    """

    PENDING = "pending"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"
    QUARANTINED = "quarantined"


# Per-stage retry limits.
# Set conservatively: GPU-heavy stages (ocr_vision) get fewer attempts to
# avoid burning GPU time on a fundamentally broken chunk.
MAX_ATTEMPTS_BY_STAGE: dict[ChunkStage, int] = {
    ChunkStage.SCENE: 3,       # Deterministic ffmpeg-based, cheap
    ChunkStage.OCR_VISION: 2,  # VLM inference — cap cost before quarantine
    ChunkStage.SPEECH: 3,      # Whisper CPU-bound, deterministic
    ChunkStage.AUDIO_EVENT: 3, # CLAP is cheap
    ChunkStage.METADATA: 5,    # Near-zero cost, rarely fails
    ChunkStage.FUSION: 3,      # Pure data assembly
}

# Fan-in base: scene/ocr_vision count as 1 track (chained), speech = 1.
# Dispatcher adds 1 for audio_event if enabled, 1 for metadata.
FANIN_BASE_COUNT = 2  # scene-chain + speech


def make_chunk_id(media_id: str, chunk_index: int, source_sha256: str) -> str:
    """Return the stable identity hash for one time window of one file.

    IMPORTANT: stage_version and model version are NOT included here.
    Including them would produce different chunk_ids for the same time
    window across stages, breaking FusionAgent's ability to join them.

    A version bump re-runs the same chunk_id with a new stage_version row.

    Args:
        media_id: Unique identifier for the media file.
        chunk_index: 0-based index of the time window.
        source_sha256: SHA-256 hex digest of the source file bytes.
            Ensures re-upload of a different file with the same name
            produces a new chunk_id, not a reuse of stale data.

    Returns:
        64-character lowercase hex SHA-256 digest.
    """
    raw = f"{media_id}:{chunk_index}:{source_sha256}".encode()
    return hashlib.sha256(raw).hexdigest()


def compute_chunk_boundaries(
    total_duration_s: float,
    chunk_duration_s: float = 600.0,
) -> list[tuple[float, float]]:
    """Return (start_s, end_s) pairs for all chunks covering the media.

    Boundaries are in seconds from the start of the video.
    Timestamps stored in TimelineEvent are in absolute milliseconds
    from video start — NOT relative to the chunk. The chunk is only a
    processing window; it is discarded after ingestion.

    Args:
        total_duration_s: Total media duration in seconds.
        chunk_duration_s: Target window size in seconds (default 10 min).

    Returns:
        List of (chunk_start_s, chunk_end_s) tuples, in order.
        Last tuple's end_s equals total_duration_s exactly.
    """
    if chunk_duration_s <= 0:
        raise ValueError(f"chunk_duration_s must be > 0, got {chunk_duration_s}")
    if total_duration_s <= 0:
        return [(0.0, 0.0)]

    boundaries: list[tuple[float, float]] = []
    start = 0.0
    while start < total_duration_s:
        end = min(start + chunk_duration_s, total_duration_s)
        boundaries.append((start, end))
        start = end
    return boundaries
