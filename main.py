"""Command-line entrypoint for the media ingestion pipeline.

This module provides a simple executable interface for processing a single
video file through the `IngestionPipeline`. It validates input paths,
sets up platform-appropriate asyncio policies, and triggers asynchronous
processing of the video, including transcription, frame analysis, vector
embedding, and Qdrant indexing.

Usage:
    uv run python main.py <path_to_video>
"""

from __future__ import annotations

import asyncio
import sys
from pathlib import Path

from core.ingestion.pipeline import IngestionPipeline


async def _run(video_path: Path) -> None:
    """Run the ingestion pipeline for a single video.

    Args:
        video_path: Path to the video file that should be processed.
    """
    pipeline = IngestionPipeline(
        qdrant_backend="docker",
        qdrant_host="localhost",
        qdrant_port=6333,
        frame_interval_seconds=5,
    )
    await pipeline.process_video(video_path)


def main() -> None:
    """CLI entrypoint for the media ingestion pipeline.

    Usage:
        uv run python main.py <path_to_video>

    The command will:

      * Validate that the given path exists and is a file.
      * Initialize the ingestion pipeline.
      * Run asynchronous processing on the specified video.
    """
    if len(sys.argv) < 2:
        print("Usage: uv run python main.py <path_to_video>")
        raise SystemExit(1)

    raw_path = sys.argv[1]
    video_path = Path(raw_path).expanduser().resolve()

    if not video_path.exists() or not video_path.is_file():
        print(f"Error: File not found or not a file: {video_path}")
        raise SystemExit(1)

    # Windows asyncio event-loop policy adjustment (if needed).
    if sys.platform == "win32":  # pragma: no cover - platform-specific
        asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())

    asyncio.run(_run(video_path))


if __name__ == "__main__":
    main()
