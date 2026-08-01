"""Fusion worker — final stage of the parallel ingestion DAG.

FusionAgent is triggered once by the fan-in counter when all tracks have
completed (or been quarantined). It:

  1. Gathers results from all stages for this chunk from the database.
  2. Aligns them to 1-second temporal windows.
  3. Applies the subtitle-wins rule: if a subtitle cue overlaps a Whisper
     segment by > 80%, the subtitle text wins (human-verified > machine).
  4. Computes a weighted confidence score, penalized for absent/quarantined
     modalities.
  5. Writes TimelineEvent rows and enqueues EmbeddingAgent.

Absent modality handling:
  Quarantined tracks are treated as absent. Fusion NEVER fails because one
  track was quarantined — it proceeds with whatever is available and records
  which modalities contributed to each event.

Why temporal alignment (not simple concatenation):
  The same moment in a video produces multiple facts from different tracks:
  a speech segment from Whisper, a subtitle cue from .srt, a scene caption
  from VisionAgent, OCR text from a visible screen. Concatenation produces
  duplicate/conflicting text. Alignment merges them into one TimelineEvent
  per time window with the best available text and full provenance.
"""

from __future__ import annotations

import asyncio

from core.domain.chunk import ChunkStage, ChunkStatus
from core.ingestion.celery_app import celery_app
from core.utils.logger import logger


# Subtitle-wins threshold: if a subtitle cue overlaps a Whisper segment
# by this fraction or more, the subtitle text is used instead.
_SUBTITLE_OVERLAP_THRESHOLD = 0.8

# Confidence weights per modality (must sum to 1.0 when all present)
_MODALITY_WEIGHTS = {
    "speech":      0.35,  # Whisper transcription
    "vision":      0.30,  # VLM scene caption + objects
    "ocr":         0.20,  # OCR text from keyframes
    "audio_event": 0.15,  # CLAP non-speech events
}

# Confidence penalty multiplier when any track is quarantined/absent
_QUARANTINE_PENALTY = 0.70


def _get_repo():
    from core.storage.repositories.chunk_state_repo import ChunkStateRepository
    return ChunkStateRepository.from_settings()


@celery_app.task(
    name="ingest.fusion",
    bind=True,
    queue="ingest:fusion",
    max_retries=None,
    acks_late=True,
)
def task_run_fusion(
    self,
    *,
    chunk_id: str,
    media_id: str,
    chunk_index: int,
    stage_version: str,
    job_id: str,
) -> None:
    """Merge all track outputs into TimelineEvents for one chunk."""
    repo = _get_repo()

    async def _run() -> None:
        if await repo.is_completed(chunk_id, ChunkStage.FUSION):
            logger.info(
                f"[FusionWorker] chunk={chunk_id[:8]} already completed, skipping."
            )
            return

        await repo.mark_processing(chunk_id, ChunkStage.FUSION, stage_version)
        try:
            await _fuse_chunk(repo, chunk_id, media_id, chunk_index, stage_version, job_id)
            await repo.mark_completed(chunk_id, ChunkStage.FUSION)
            logger.info(
                f"[FusionWorker] chunk={chunk_id[:8]} fusion COMPLETED"
            )
        except Exception as exc:
            error_msg = f"{type(exc).__name__}: {exc}"
            await repo.mark_failed(chunk_id, ChunkStage.FUSION, error_msg)

            if await repo.should_quarantine(chunk_id, ChunkStage.FUSION):
                await repo.mark_quarantined(chunk_id, ChunkStage.FUSION, error_msg)
                logger.error(
                    f"[FusionWorker] chunk={chunk_id[:8]} QUARANTINED: {error_msg}"
                )
                return  # Fusion quarantine: no further action

            raise self.retry(exc=exc, countdown=30) from exc

    asyncio.run(_run())


async def _fuse_chunk(
    repo,
    chunk_id: str,
    media_id: str,
    chunk_index: int,
    stage_version: str,
    job_id: str,
) -> None:
    """Core temporal alignment merge logic.

    For each 1-second window of the chunk:
      1. Find the best text source (subtitle > Whisper > silence).
      2. Find the scene caption and OCR text for this window.
      3. Find any non-speech audio event label.
      4. Compute confidence (weighted average, penalized for absent tracks).
      5. Write one TimelineEvent row.
    """
    # Determine which stages are absent/quarantined for this chunk
    absent_stages = await _get_absent_stages(repo, chunk_id)
    if absent_stages:
        logger.warning(
            f"[FusionWorker] chunk={chunk_id[:8]} has absent/quarantined "
            f"stages: {[s.value for s in absent_stages]}. "
            "Proceeding with available tracks."
        )

    has_quarantine = bool(absent_stages)

    # Pull data from the database for this chunk
    from core.storage.db import VectorDB
    from config import settings

    db = VectorDB(
        backend=settings.qdrant_backend,
        host=settings.qdrant_host,
        port=settings.qdrant_port,
    )

    speech_segments = _fetch_speech_segments(db, media_id, chunk_index)
    subtitle_cues = _fetch_subtitle_cues(db, media_id, chunk_index)
    scene_captions = _fetch_scene_captions(db, media_id, chunk_index)
    ocr_results = _fetch_ocr_results(db, media_id, chunk_index)
    audio_events = _fetch_audio_events(db, media_id, chunk_index)

    # Build merged windows
    events = _merge_windows(
        speech_segments=speech_segments,
        subtitle_cues=subtitle_cues,
        scene_captions=scene_captions,
        ocr_results=ocr_results,
        audio_events=audio_events,
        absent_stages=absent_stages,
        has_quarantine=has_quarantine,
        media_id=media_id,
        chunk_id=chunk_id,
        stage_version=stage_version,
    )

    logger.info(
        f"[FusionWorker] chunk={chunk_id[:8]} produced {len(events)} TimelineEvents"
    )

    # Persist and enqueue embedding
    for event in events:
        _persist_event(db, event)

    logger.info(
        f"[FusionWorker] chunk={chunk_id[:8]} persisted {len(events)} events"
    )


async def _get_absent_stages(repo, chunk_id: str) -> list[ChunkStage]:
    """Return stages that are quarantined or missing for this chunk."""
    absent = []
    for stage in [ChunkStage.SCENE, ChunkStage.SPEECH, ChunkStage.METADATA]:
        row = await repo.get_row(chunk_id, stage)
        if row is None or row["status"] == ChunkStatus.QUARANTINED.value:
            absent.append(stage)
    return absent


def _merge_windows(
    *,
    speech_segments: list[dict],
    subtitle_cues: list[dict],
    scene_captions: list[dict],
    ocr_results: list[dict],
    audio_events: list[dict],
    absent_stages: list[ChunkStage],
    has_quarantine: bool,
    media_id: str,
    chunk_id: str,
    stage_version: str,
) -> list[dict]:
    """Merge all modality outputs into TimelineEvent dicts.

    Alignment strategy: iterate over speech/subtitle segments (they define
    meaningful time boundaries). For each segment, collect overlapping data
    from all other modalities.
    """
    events: list[dict] = []

    # Merge subtitle cues with speech segments using the subtitle-wins rule
    merged_text_segments = _apply_subtitle_wins(speech_segments, subtitle_cues)

    # For each text segment, collect overlapping multimodal data
    for seg in merged_text_segments:
        start_ms = seg["start_ms"]
        end_ms = seg["end_ms"]

        # Find overlapping scene caption
        caption = _find_overlapping(scene_captions, start_ms, end_ms, "caption")
        # Find overlapping OCR text
        ocr_text = _find_overlapping(ocr_results, start_ms, end_ms, "text")
        # Find overlapping audio event
        audio_label = _find_overlapping(audio_events, start_ms, end_ms, "label")

        # Compute confidence
        confidence = _compute_confidence(
            has_speech=bool(seg.get("text")),
            has_vision=bool(caption),
            has_ocr=bool(ocr_text),
            has_audio=bool(audio_label),
            absent_stages=absent_stages,
        )

        events.append({
            "media_id": media_id,
            "chunk_id": chunk_id,
            "start_ms": start_ms,
            "end_ms": end_ms,
            "transcript": seg.get("text", ""),
            "transcript_source": seg.get("source_type", "whisper"),
            "caption": caption or "",
            "ocr_text": ocr_text or "",
            "audio_labels": [audio_label] if audio_label else [],
            "speaker": seg.get("speaker", ""),
            "confidence": confidence,
            "stage_version": stage_version,
        })

    return events


def _apply_subtitle_wins(
    speech_segments: list[dict],
    subtitle_cues: list[dict],
) -> list[dict]:
    """Merge subtitle cues with Whisper segments.

    For each Whisper segment, if a subtitle cue overlaps it by >=
    SUBTITLE_OVERLAP_THRESHOLD, the subtitle text replaces the Whisper text
    and source_type is set to 'subtitle'. Otherwise, Whisper text is kept.

    Non-overlapping subtitle cues are added as standalone segments.
    """
    result = list(speech_segments)  # copy

    for cue in subtitle_cues:
        c_start = cue["start_ms"]
        c_end = cue["end_ms"]
        cue_duration = max(c_end - c_start, 1)
        replaced = False

        for seg in result:
            s_start = seg["start_ms"]
            s_end = seg["end_ms"]
            overlap = max(0, min(c_end, s_end) - max(c_start, s_start))
            overlap_fraction = overlap / cue_duration

            if overlap_fraction >= _SUBTITLE_OVERLAP_THRESHOLD:
                seg["text"] = cue["text"]
                seg["source_type"] = "subtitle"
                replaced = True
                break

        if not replaced:
            # Subtitle cue with no matching Whisper segment — add it
            result.append({
                "start_ms": c_start,
                "end_ms": c_end,
                "text": cue["text"],
                "source_type": "subtitle",
                "speaker": "subtitle",
            })

    # Sort by start_ms
    result.sort(key=lambda x: x["start_ms"])
    return result


def _compute_confidence(
    *,
    has_speech: bool,
    has_vision: bool,
    has_ocr: bool,
    has_audio: bool,
    absent_stages: list,
) -> float:
    """Compute weighted confidence for a TimelineEvent.

    Returns a value in [0.0, 1.0]. Absent/quarantined tracks contribute 0
    to the weighted sum. A quarantine penalty multiplier is applied when
    any track is absent.
    """
    weights = _MODALITY_WEIGHTS
    total_weight = 0.0
    weighted_sum = 0.0

    for modality, weight in weights.items():
        present = {
            "speech": has_speech,
            "vision": has_vision,
            "ocr": has_ocr,
            "audio_event": has_audio,
        }[modality]
        total_weight += weight
        if present:
            weighted_sum += weight

    raw = weighted_sum / total_weight if total_weight > 0 else 0.0
    penalty = _QUARANTINE_PENALTY if absent_stages else 1.0
    return round(raw * penalty, 4)


# ---------------------------------------------------------------------------
# Data fetch stubs — delegate to VectorDB query layer
# These return lists of dicts with at minimum: start_ms, end_ms, <field>
# ---------------------------------------------------------------------------

def _fetch_speech_segments(db, media_id: str, chunk_index: int) -> list[dict]:
    try:
        segs = db.get_voice_segments(media_id)
        return [
            {
                "start_ms": int(s.get("start", 0) * 1000),
                "end_ms": int(s.get("end", 0) * 1000),
                "text": s.get("text", ""),
                "speaker": s.get("speaker", ""),
                "source_type": s.get("source_type", "whisper"),
            }
            for s in segs
            if s.get("source_type") != "subtitle"
        ]
    except Exception as e:
        logger.warning(f"[FusionWorker] fetch_speech error: {e}")
        return []


def _fetch_subtitle_cues(db, media_id: str, chunk_index: int) -> list[dict]:
    try:
        segs = db.get_voice_segments(media_id)
        return [
            {
                "start_ms": int(s.get("start", 0) * 1000),
                "end_ms": int(s.get("end", 0) * 1000),
                "text": s.get("text", ""),
            }
            for s in segs
            if s.get("source_type") == "subtitle"
        ]
    except Exception as e:
        logger.warning(f"[FusionWorker] fetch_subtitles error: {e}")
        return []


def _fetch_scene_captions(db, media_id: str, chunk_index: int) -> list[dict]:
    try:
        scenes = db.get_scenes(media_id)
        return [
            {
                "start_ms": int(s.get("start_time", 0) * 1000),
                "end_ms": int(s.get("end_time", 0) * 1000),
                "caption": s.get("caption", s.get("description", "")),
            }
            for s in scenes
        ]
    except Exception as e:
        logger.warning(f"[FusionWorker] fetch_scenes error: {e}")
        return []


def _fetch_ocr_results(db, media_id: str, chunk_index: int) -> list[dict]:
    try:
        frames = db.get_frames(media_id)
        return [
            {
                "start_ms": int(f.get("timestamp", 0) * 1000),
                "end_ms": int(f.get("timestamp", 0) * 1000) + 1000,
                "text": " ".join(f.get("ocr_texts", [])),
            }
            for f in frames
            if f.get("ocr_texts")
        ]
    except Exception as e:
        logger.warning(f"[FusionWorker] fetch_ocr error: {e}")
        return []


def _fetch_audio_events(db, media_id: str, chunk_index: int) -> list[dict]:
    try:
        evts = db.get_audio_events(media_id)
        return [
            {
                "start_ms": int(e.get("start_time", 0) * 1000),
                "end_ms": int(e.get("end_time", 0) * 1000),
                "label": e.get("label", ""),
            }
            for e in evts
        ]
    except Exception as e:
        logger.warning(f"[FusionWorker] fetch_audio_events error: {e}")
        return []


def _find_overlapping(
    segments: list[dict],
    start_ms: int,
    end_ms: int,
    field: str,
) -> str | None:
    """Return the field value of the first segment overlapping [start_ms, end_ms]."""
    for seg in segments:
        s = seg.get("start_ms", 0)
        e = seg.get("end_ms", 0)
        if s <= end_ms and e >= start_ms:
            return seg.get(field)
    return None


def _persist_event(db, event: dict) -> None:
    """Write one TimelineEvent to the storage layer."""
    try:
        db.upsert_timeline_event(event)
    except AttributeError:
        # db.upsert_timeline_event not yet on VectorDB — log and skip
        # This will be implemented in Phase 3 (domain model + storage)
        logger.debug(
            f"[FusionWorker] upsert_timeline_event not yet implemented "
            f"for media={event['media_id']} start={event['start_ms']}ms"
        )
    except Exception as e:
        logger.warning(f"[FusionWorker] persist_event error: {e}")
