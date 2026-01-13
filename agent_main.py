"""Antigravity Agent CLI Entry Point.

Usage:
    python agent_main.py video.mp4 --task="analyze"
    python agent_main.py video.mp4 --task="remove the red car"
"""

import argparse
import sys
from pathlib import Path

from config import settings
from core.orchestration.orchestrator import get_orchestrator


async def main_async() -> None:
    """Async entry point for the agent CLI."""
    parser = argparse.ArgumentParser(
        description="AI Media Agent (Antigravity)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("video_path", type=str, help="Path to input video")
    parser.add_argument(
        "--task",
        type=str,
        default="analyze",
        help="Natural language task description",
    )

    args = parser.parse_args()

    video = Path(args.video_path)
    if not video.exists():
        print(f"Error: {video} does not exist.")
        sys.exit(1)

    print("--- Antigravity Agent ---")
    print(f"Video: {video.name}")
    print(f"Task: {args.task}")
    print(f"SAM3: {settings.enable_sam3_tracking}")
    print(f"IndicASR: {settings.use_indic_asr}")
    print(f"Manipulation: {settings.manipulation_backend}")
    print()

    # Initialize Orchestrator
    try:
        orchestrator = get_orchestrator()
        # In a real scenario, we might want to register the video with the agent context first
        # But for now, we pass the task directly.
        # Construct a query that includes video context if possible, or just the task.
        query = f"For video '{video.name}': {args.task}"

        print(f"Executing Query: {query}")
        result = await orchestrator.execute(query)

        print()
        print("--- Result ---")
        if "results" in result:
            print(f"Found {len(result['results'])} matches")
            for res in result["results"]:
                print(f"- {res}")
        else:
            print(result)

    except Exception as e:
        print(f"Error executing agent: {e}")
        import traceback

        traceback.print_exc()


def main():
    """Sync wrapper for the async main entry point."""
    import asyncio

    asyncio.run(main_async())


if __name__ == "__main__":
    main()
