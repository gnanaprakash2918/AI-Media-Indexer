"""Metadata worker — Track 4 of the parallel ingestion DAG.

Runs MetadataAgent: reads container/sidecar metadata, filename hints, TMDB
lookups, and — critically — parses subtitle files (.srt, .ass, .vtt) as a
first-class indexed modality.

Why subtitles are in MetadataAgent (not SpeechAgent):
  Subtitles are file-system artifacts, not audio-derived. They are read
  the same way as .nfo or TMDB data — from a sidecar file alongside the
  media. However, their cues are aligned with speech timestamps, so
  FusionAgent merges them with Whisper output using the subtitle-wins rule:
    - If a subtitle cue overlaps a Whisper segment by > 80%, the subtitle
      text wins (human-verified > machine-transcribed).
    - EvidenceRef.source_type = 'subtitle' | 'whisper' is set accordingly.

Runs in parallel with scene and speech — no dependency on keyframes or
transcription. Typically completes in seconds.
"""

from __future__ import annotations

import asyncio
from pathlib import Path

from core.domain.chunk import ChunkStage
from core.ingestion.celery_app import celery_app
from core.ingestion.workers.fanin import increment_and_check
from core.utils.logger import logger


def _get_repo():
    from core.storage.repositories.chunk_state_repo import ChunkStateRepository
    return ChunkStateRepository.from_settings()


@celery_app.task(
    name="ingest.metadata",
    bind=True,
    queue="ingest:metadata",
    max_retries=None,
    acks_late=True,
)
def task_run_metadata(
    self,
    *,
    chunk_id: str,
    media_id: str,
    chunk_index: int,
    media_path: str,
    stage_version: str,
    job_id: str,
    tmdb_hint: str | None = None,
) -> None:
    """Run MetadataAgent + subtitle parsing for one chunk."""
    repo = _get_repo()

    async def _run() -> None:
        if await repo.is_completed(chunk_id, ChunkStage.METADATA):
            logger.info(
                f"[MetadataWorker] chunk={chunk_id[:8]} already completed, skipping."
            )
            return

        await repo.mark_processing(chunk_id, ChunkStage.METADATA, stage_version)
        try:
            from core.ingestion.pipeline import IngestionPipeline

            pipeline = IngestionPipeline()
            path = Path(media_path)

            # MetadataAgent: container, filename, sidecar, TMDB lookup
            await pipeline.metadata_engine.identify(
                path, user_hint=tmdb_hint
            )

            # Subtitle parsing: read .srt/.ass/.vtt sidecars alongside media
            subtitle_cues = _parse_subtitles(path)
            if subtitle_cues:
                logger.info(
                    f"[MetadataWorker] Found {len(subtitle_cues)} subtitle cues "
                    f"for chunk={chunk_id[:8]} — stored for Fusion subtitle-wins merge."
                )
                _store_subtitle_cues(pipeline, media_id, chunk_id, subtitle_cues)

            await repo.mark_completed(chunk_id, ChunkStage.METADATA)
            logger.info(
                f"[MetadataWorker] chunk={chunk_id[:8]} metadata COMPLETED"
            )

        except Exception as exc:
            error_msg = f"{type(exc).__name__}: {exc}"
            await repo.mark_failed(chunk_id, ChunkStage.METADATA, error_msg)

            if await repo.should_quarantine(chunk_id, ChunkStage.METADATA):
                await repo.mark_quarantined(chunk_id, ChunkStage.METADATA, error_msg)
                logger.error(
                    f"[MetadataWorker] chunk={chunk_id[:8]} QUARANTINED: {error_msg}"
                )
                if increment_and_check(chunk_id):
                    _trigger_fusion(chunk_id, media_id, chunk_index, stage_version, job_id)
                return

            raise self.retry(exc=exc, countdown=15) from exc  # shorter delay: fast stage

        if increment_and_check(chunk_id):
            _trigger_fusion(chunk_id, media_id, chunk_index, stage_version, job_id)

    asyncio.run(_run())


# ---------------------------------------------------------------------------
# Subtitle parsing helpers
# ---------------------------------------------------------------------------

def _parse_subtitles(media_path: Path) -> list[dict]:
    """Find and parse subtitle sidecar files next to the media file.

    Supports: .srt (SubRip), .vtt (WebVTT), .ass/.ssa (Advanced SubStation).

    Returns list of cue dicts:
        {"start_ms": int, "end_ms": int, "text": str}
    Times are absolute ms from video start (same coordinate space as
    all other TimelineEvent timestamps).
    """
    cues: list[dict] = []
    base = media_path.with_suffix("")

    for ext in (".srt", ".vtt", ".ass", ".ssa"):
        subtitle_path = base.with_suffix(ext)
        if subtitle_path.exists():
            logger.info(f"[SubtitleParser] Found {subtitle_path.name}")
            try:
                cues.extend(_parse_file(subtitle_path, ext))
            except Exception as e:
                logger.warning(
                    f"[SubtitleParser] Failed to parse {subtitle_path.name}: {e}"
                )

    return cues


def _parse_file(path: Path, ext: str) -> list[dict]:
    """Dispatch to the correct parser by extension."""
    if ext == ".srt":
        return _parse_srt(path)
    if ext == ".vtt":
        return _parse_vtt(path)
    if ext in (".ass", ".ssa"):
        return _parse_ass(path)
    return []


def _timecode_to_ms(timecode: str) -> int:
    """Convert HH:MM:SS,mmm or HH:MM:SS.mmm to milliseconds."""
    timecode = timecode.replace(",", ".")
    parts = timecode.split(":")
    h = int(parts[0])
    m = int(parts[1])
    s_ms = parts[2].split(".")
    s = int(s_ms[0])
    ms = int(s_ms[1]) if len(s_ms) > 1 else 0
    return ((h * 3600) + (m * 60) + s) * 1000 + ms


def _parse_srt(path: Path) -> list[dict]:
    """Parse SubRip (.srt) subtitle file."""
    cues: list[dict] = []
    content = path.read_text(encoding="utf-8", errors="replace")
    blocks = content.strip().split("\n\n")
    for block in blocks:
        lines = block.strip().splitlines()
        if len(lines) < 3:
            continue
        # Line 0: sequence number (skip)
        # Line 1: timestamps "00:01:23,456 --> 00:01:26,789"
        # Line 2+: text
        try:
            ts_line = lines[1]
            start_str, end_str = ts_line.split(" --> ")
            start_ms = _timecode_to_ms(start_str.strip())
            end_ms = _timecode_to_ms(end_str.strip())
            text = " ".join(lines[2:]).strip()
            if text:
                cues.append({"start_ms": start_ms, "end_ms": end_ms, "text": text})
        except (ValueError, IndexError):
            continue
    return cues


def _parse_vtt(path: Path) -> list[dict]:
    """Parse WebVTT (.vtt) subtitle file (simplified, handles standard cues)."""
    cues: list[dict] = []
    content = path.read_text(encoding="utf-8", errors="replace")
    # Remove WEBVTT header
    blocks = content.split("\n\n")
    for block in blocks:
        lines = block.strip().splitlines()
        # Find the line with --> arrow
        ts_idx = next(
            (i for i, l in enumerate(lines) if " --> " in l), None
        )
        if ts_idx is None:
            continue
        try:
            start_str, end_str = lines[ts_idx].split(" --> ")
            start_ms = _timecode_to_ms(start_str.strip().split()[0])
            end_ms = _timecode_to_ms(end_str.strip().split()[0])
            text = " ".join(lines[ts_idx + 1:]).strip()
            if text:
                cues.append({"start_ms": start_ms, "end_ms": end_ms, "text": text})
        except (ValueError, IndexError):
            continue
    return cues


def _parse_ass(path: Path) -> list[dict]:
    """Parse ASS/SSA subtitle file (extract Dialogue lines only)."""
    cues: list[dict] = []
    content = path.read_text(encoding="utf-8", errors="replace")
    in_events = False
    format_fields: list[str] = []

    for line in content.splitlines():
        line = line.strip()
        if line == "[Events]":
            in_events = True
            continue
        if in_events and line.startswith("Format:"):
            format_fields = [f.strip() for f in line[7:].split(",")]
            continue
        if in_events and line.startswith("Dialogue:"):
            parts = line[9:].split(",", len(format_fields) - 1)
            try:
                fi = {f: parts[i] for i, f in enumerate(format_fields)}
                start_ms = _timecode_to_ms(fi.get("Start", "0:00:00.00"))
                end_ms = _timecode_to_ms(fi.get("End", "0:00:00.00"))
                # Strip ASS override tags like {\an8}
                text_raw = fi.get("Text", "")
                import re
                text = re.sub(r"\{[^}]*\}", "", text_raw).replace("\\N", " ").strip()
                if text:
                    cues.append({"start_ms": start_ms, "end_ms": end_ms, "text": text})
            except (KeyError, IndexError, ValueError):
                continue

    return cues


def _store_subtitle_cues(
    pipeline: object,
    media_id: str,
    chunk_id: str,
    cues: list[dict],
) -> None:
    """Store subtitle cues as provisional speech segments tagged source_type='subtitle'.

    FusionAgent later applies the subtitle-wins rule: if a subtitle cue
    overlaps a Whisper segment by > 80%, the subtitle text replaces the
    Whisper text and EvidenceRef.source_type = 'subtitle'.
    """
    db = getattr(pipeline, "db", None)
    if db is None:
        return

    for cue in cues:
        try:
            db.upsert_voice_segment(
                media_path=media_id,
                start=cue["start_ms"] / 1000.0,
                end=cue["end_ms"] / 1000.0,
                text=cue["text"],
                speaker="subtitle",
                source_type="subtitle",
                chunk_id=chunk_id,
            )
        except Exception as e:
            logger.debug(f"[SubtitleStore] Could not store cue: {e}")


def _trigger_fusion(
    chunk_id: str,
    media_id: str,
    chunk_index: int,
    stage_version: str,
    job_id: str,
) -> None:
    from core.ingestion.workers.fusion_worker import task_run_fusion

    logger.info(
        f"[FanIn] All tracks done for chunk={chunk_id[:8]}. Triggering fusion."
    )
    task_run_fusion.apply_async(
        kwargs={
            "chunk_id": chunk_id,
            "media_id": media_id,
            "chunk_index": chunk_index,
            "stage_version": stage_version,
            "job_id": job_id,
        },
        queue="ingest:fusion",
    )
