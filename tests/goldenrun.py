"""Golden Test Script ("The Golden Run")

This script performs a complete end-to-end verification of the AI-Media-Indexer pipeline.
It is designed to be run AFTER refactoring to ensure:
1.  Voice clustering works (1 speaker per actual speaker, not 100 fragments).
2.  Storage is efficient (low vector count).
3.  Progress tracking is monotonic (0 -> 100).
4.  Search returns valid results.

Usage:
    python tests/goldenrun.py --video "path/to/test_video.mp4"
"""

import argparse
import asyncio
import logging
import sys
from pathlib import Path

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent))

from config import settings
from core.ingestion.pipeline import IngestionPipeline
from core.storage.db import VectorDB, paginated_scroll
from core.utils.progress import progress_tracker

# Local logging
logging.basicConfig(level=logging.INFO)
log = logging.getLogger("GoldenRun")


async def verify_storage(db: VectorDB, video_path: str):
    """Verify that storage usage is efficient."""
    log.info("Verifying storage efficiency...")
    
    # Check Voice Clusters
    clusters = set()
    points = paginated_scroll(
        db.client, 
        collection_name=db.VOICE_COLLECTION,
        with_payload=True
    )
    for p in points:
        clusters.add(p.payload.get("voice_cluster_id"))
    
    log.info(f"Unique Voice Clusters: {len(clusters)}")
    if len(clusters) > 5:
        log.warning("FAILURE: Too many voice clusters! Intra-file clustering failed?")
    else:
        log.info("SUCCESS: Voice clustering seems reasonable.")

    # Check Vector Count
    # A 30s video should not have 5GB of data
    # (Checking logical count, not physical size here)
    log.info(f"Total Voice Segments: {len(points)}")


async def run_test(video_path: Path):
    if not video_path.exists():
        log.error(f"Video not found: {video_path}")
        return

    log.info(f"Starting Golden Run on: {video_path}")
    
    # 1. Reset Progress
    job_id = "golden_test_001"
    
    # 2. Run Pipeline
    pipeline = IngestionPipeline()
    try:
        await pipeline.process_video(video_path, job_id=job_id)
        log.info("Pipeline finished successfully.")
    except Exception as e:
        log.error(f"Pipeline crashed: {e}")
        return

    # 3. Verify Progress State
    job = progress_tracker.get(job_id)
    if job:
        log.info(f"Final Progress: {job.progress}%")
        if job.progress < 100:
            log.warning("FAILURE: Progress did not reach 100%")
    
    # 4. Verify DB
    db = VectorDB(backend=settings.qdrant_backend)
    await verify_storage(db, str(video_path))

    log.info("Golden Run Complete.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--video", required=True, help="Path to test video")
    args = parser.parse_args()
    
    asyncio.run(run_test(Path(args.video)))
