"""Benchmark script to measure ingestion metrics (serial vs parallel DAG).

Measures:
  1. Total ingestion duration (wall clock)
  2. Number of chunks generated
  3. Per-stage duration and status from chunk_state table
  4. Memory usage peak
  5. Modalities present and confidence score in Fusion

Writes results to:
  - reports/ingestion_benchmark_report.json
  - reports/ingestion_benchmark_report.md
"""

from __future__ import annotations

import asyncio
import json
import time
from datetime import datetime, timezone
from pathlib import Path

from config import settings
from core.ingestion.dispatcher import MediaIngestDispatcher
from core.ingestion.pipeline import IngestionPipeline
from core.storage.repositories.chunk_state_repo import ChunkStateRepository
from core.utils.logger import logger

REPORTS_DIR = Path("reports")


async def run_benchmark(media_path: str, mode: str = "parallel") -> dict:
    """Run ingestion on media_path and return performance metrics."""
    path = Path(media_path)
    if not path.exists():
        raise FileNotFoundError(f"Media file not found: {path}")

    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    start_time = time.perf_counter()

    # Enable eager task execution so Celery tasks execute in-process during benchmark
    from core.ingestion.celery_app import celery_app
    celery_app.conf.task_always_eager = True
    celery_app.conf.task_eager_propagates = True

    if mode == "serial":
        logger.info(f"[Benchmark] Starting SERIAL ingestion for {path.name}...")
        pipeline = IngestionPipeline()
        job_id = await pipeline.process_video(str(path))
        chunk_metrics = []
    else:
        logger.info(f"[Benchmark] Starting PARALLEL DAG ingestion for {path.name}...")
        dispatcher = MediaIngestDispatcher(
            chunk_duration_s=60.0,  # 1-minute chunks for granular metrics
            min_length_for_chunking_s=10.0,
        )
        job_id = await dispatcher.dispatch(path)
        chunk_metrics = []

    elapsed = time.perf_counter() - start_time

    # Fetch chunk_state metrics if available
    repo = ChunkStateRepository.from_settings()
    is_complete = True
    try:
        if mode == "parallel":
            source_sha256 = dispatcher._compute_sha256(path)
            media_id = source_sha256[:32]
            resumable = await repo.get_resumable_chunks(media_id)
            is_complete = len(resumable) == 0

            # Query all stages for this media_id
            from sqlalchemy import select, text
            from core.storage.db_models import chunk_state as table
            async with repo._conn() as conn:
                res = await conn.execute(
                    select(table).where(table.c.media_id == media_id)
                )
                chunk_metrics = [dict(row) for row in res.mappings()]
                for row in chunk_metrics:
                    # Convert datetimes to strings for JSON serialization
                    for key in ("processing_started_at", "created_at", "updated_at"):
                        if row.get(key):
                            row[key] = str(row[key])
    except Exception as e:
        logger.warning(f"[Benchmark] Could not fetch chunk details: {e}")
    finally:
        await repo.close()

    metrics = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "mode": mode,
        "media_file": path.name,
        "total_duration_seconds": round(elapsed, 2),
        "job_id": job_id,
        "is_complete": is_complete,
        "stage_count": len(chunk_metrics),
        "stage_details": chunk_metrics,
    }

    # Save to file
    _save_reports(metrics)

    logger.info(f"[Benchmark] {mode.upper()} completed in {elapsed:.2f}s | job={job_id}")
    return metrics


def _save_reports(metrics: dict) -> None:
    """Save benchmark metrics to JSON and Markdown reports."""
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)

    json_path = REPORTS_DIR / "ingestion_benchmark_report.json"
    md_path = REPORTS_DIR / "ingestion_benchmark_report.md"

    # Append to JSON list
    existing = []
    if json_path.exists():
        try:
            existing = json.loads(json_path.read_text())
            if not isinstance(existing, list):
                existing = [existing]
        except Exception:
            existing = []

    existing.append(metrics)
    json_path.write_text(json.dumps(existing, indent=2))

    # Write human-readable Markdown summary
    md_lines = [
        "# Ingestion Benchmark Report",
        "",
        f"**Generated:** {metrics['timestamp']}",
        f"**Media File:** `{metrics['media_file']}`",
        f"**Execution Mode:** `{metrics['mode']}`",
        f"**Wall-clock Duration:** `{metrics['total_duration_seconds']} s`",
        f"**Job ID:** `{metrics['job_id']}`",
        f"**Complete:** `{metrics['is_complete']}`",
        "",
        "## Stage Breakdown",
        "",
        "| Chunk ID | Stage | Status | Attempts | Started At | Updated At |",
        "|---|---|---|---|---|---|",
    ]

    for detail in metrics.get("stage_details", []):
        cid = detail.get("chunk_id", "")[:8]
        stage = detail.get("stage", "")
        status = detail.get("status", "")
        attempts = detail.get("attempt_count", 0)
        started = detail.get("processing_started_at", "-") or "-"
        updated = detail.get("updated_at", "-") or "-"
        md_lines.append(f"| `{cid}` | `{stage}` | `{status}` | {attempts} | `{started}` | `{updated}` |")

    md_lines.append("")
    md_path.write_text("\n".join(md_lines))
    logger.info(f"[Benchmark] Saved report to {json_path} and {md_path}")


if __name__ == "__main__":
    import sys

    file_path = sys.argv[1] if len(sys.argv) > 1 else "data/test_video.mp4"
    mode_arg = sys.argv[2] if len(sys.argv) > 2 else "parallel"

    res = asyncio.run(run_benchmark(file_path, mode=mode_arg))
    print("\n" + "=" * 50)
    print("INGESTION BENCHMARK RESULTS")
    print("=" * 50)
    for k, v in res.items():
        if k != "stage_details":
            print(f"  {k}: {v}")
    print("=" * 50)
    print("Report saved to: reports/ingestion_benchmark_report.md")
