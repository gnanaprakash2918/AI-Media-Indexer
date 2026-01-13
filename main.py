"""Command-line entrypoint for the media ingestion pipeline.

This module provides a simple executable interface for processing a single
video file through the `IngestionPipeline`. It validates input paths,
supports interactive filename resolution, lets the user provide a media
type hint, sets up platform-appropriate asyncio policies, and triggers
asynchronous processing of the video, including transcription, frame
analysis, vector embedding, and Qdrant indexing.

Usage:
    uv run python main.py <path_to_video>
"""

from __future__ import annotations

import asyncio
import difflib
import sys
from pathlib import Path

from core.ingestion.pipeline import IngestionPipeline


def _ask_media_type() -> str:
    """Interactively ask the user for a media type hint.

    Returns:
        A lowercase string corresponding to a `MediaType` value, such as
        ``"movie"``, ``"tv"``, ``"personal"``, or ``"unknown"``.
    """
    print("\n--- Media Type Classification ---")
    print("1. Movie (Fetch Cast/Subs)")
    print("2. TV Series (Fetch Cast/Subs)")
    print("3. Personal Video (Skip external APIs)")
    print("4. Auto-Detect (Default)")

    choice = input("Select type (1-4) [4]: ").strip()

    if choice == "1":
        return "movie"
    if choice == "2":
        return "tv"
    if choice == "3":
        return "personal"
    return "unknown"


async def _run(video_path: Path, media_type: str) -> None:
    """Run the ingestion pipeline for a single video.

    Args:
        video_path: Path to the video file that should be processed.
        media_type: Media type hint string to be forwarded to the ingestion
            pipeline. Should align with values in :class:`MediaType`, such
            as ``"movie"``, ``"tv"``, ``"personal"``, or ``"unknown"``.
    """
    pipeline = IngestionPipeline(
        qdrant_backend="docker",
        qdrant_host="localhost",
        qdrant_port=6333,
        frame_interval_seconds=15,
        # tmdb_api_key="YOUR_KEY_HERE",  # Optionally provide TMDB key.
    )
    await pipeline.process_video(video_path, media_type_hint=media_type)


def _interactive_resolve(raw_path: str) -> Path:
    """Resolve a video path, interactively offering close matches if needed.

    If the exact file is not found, this function scans the parent directory
    for likely video file candidates and lets the user pick one.

    Args:
        raw_path: Raw path string, potentially quoted, as passed from the CLI.

    Returns:
        A resolved :class:`Path` pointing to an existing video file.

    Raises:
        SystemExit: If no suitable file can be resolved or selected.
    """
    clean_path_str = raw_path.strip('"').strip("'")
    path = Path(clean_path_str).expanduser().resolve()

    if path.exists() and path.is_file():
        return path

    print(f"\n[WARN] Exact match failed for: '{path.name}'")

    parent = path.parent
    if not parent.exists():
        print(f"[ERROR] Parent directory does not exist: {parent}")
        sys.exit(1)

    print(f"[INFO] Scanning '{parent}' for candidates...")

    valid_extensions = {".mp4", ".mkv", ".avi", ".mov", ".webm"}
    candidates = [
        p
        for p in parent.iterdir()
        if p.is_file() and p.suffix.lower() in valid_extensions
    ]

    if not candidates:
        print("[ERROR] No video files found in that directory.")
        sys.exit(1)

    matches = difflib.get_close_matches(
        path.name, [p.name for p in candidates], n=5, cutoff=0.3
    )

    selection_pool = (
        [parent / m for m in matches] if matches else candidates[:10]
    )

    print("\n--- Did you mean one of these? ---")
    for i, candidate in enumerate(selection_pool):
        print(f"{i + 1}: {candidate.name}")
    print("-------------------------------------")

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

    media_type_str = _ask_media_type()

    if sys.platform == "win32":
        asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())

    asyncio.run(_run(video_path, media_type_str))


if __name__ == "__main__":
    main()
