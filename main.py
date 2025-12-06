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
import difflib
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
        frame_interval_seconds=15,
    )
    await pipeline.process_video(video_path)


def _interactive_resolve(raw_path: str) -> Path:
    """If the exact file isn't found, lists files in the directory and let user pick."""
    # 1. Clean the input
    clean_path_str = raw_path.strip('"').strip("'")
    path = Path(clean_path_str).expanduser().resolve()

    # 2. Check if it exists exactly
    if path.exists() and path.is_file():
        return path

    print(f"\n[WARN] Exact match failed for: '{path.name}'")

    # 3. Scan the parent directory
    parent = path.parent
    if not parent.exists():
        print(f"[ERROR] Parent directory does not exist: {parent}")
        sys.exit(1)

    print(f"[INFO] Scanning '{parent}' for candidates...")

    # Get all video files in that folder
    valid_extensions = {".mp4", ".mkv", ".avi", ".mov", ".webm"}
    candidates = [
        p
        for p in parent.iterdir()
        if p.is_file() and p.suffix.lower() in valid_extensions
    ]

    if not candidates:
        print("[ERROR] No video files found in that directory.")
        sys.exit(1)

    # 4. Find close matches
    matches = difflib.get_close_matches(
        path.name, [p.name for p in candidates], n=5, cutoff=0.3
    )

    selection_pool = [parent / m for m in matches] if matches else candidates[:10]

    print("\n--- Did you mean one of these? ---")
    for i, candidate in enumerate(selection_pool):
        print(f"{i + 1}: {candidate.name}")
    print("-------------------------------------")

    # 5. Ask User
    try:
        choice = input(
            f"Select a number (1-{len(selection_pool)}) or press Enter to cancel: "
        )
        if choice.strip().isdigit():
            idx = int(choice) - 1
            if 0 <= idx < len(selection_pool):
                selected = selection_pool[idx]
                print(f"[SUCCESS] Selected: {selected.name}")
                return selected
    except KeyboardInterrupt:
        pass

    print("[ERROR] No file selected.")
    sys.exit(1)


def main() -> None:
    """CLI entrypoint for the media ingestion pipeline.

    Usage:
        uv run python main.py <path_to_video>
    """
    if len(sys.argv) < 2:
        print("Usage: uv run python main.py <path_to_video>")
        raise SystemExit(1)

    raw_input = " ".join(sys.argv[1:])

    video_path = _interactive_resolve(raw_input)

    # Windows Asyncio Fix
    if sys.platform == "win32":
        asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())

    asyncio.run(_run(video_path))


if __name__ == "__main__":
    main()
