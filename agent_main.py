"""Antigravity Agent CLI Entry Point.

Usage:
    python agent_main.py video.mp4 --task="analyze"
    python agent_main.py video.mp4 --task="remove the red car"
"""

import argparse
import sys
from pathlib import Path

from config import settings
from core.orchestration.agent_graph import MediaPipelineAgent


def main() -> None:
    parser = argparse.ArgumentParser(
        description="AI Media Agent (Antigravity)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("video_path", type=str, help="Path to input video")
    parser.add_argument(
        "--task", 
        type=str, 
        default="analyze", 
        help="Natural language task description"
    )
    
    args = parser.parse_args()
    
    video = Path(args.video_path)
    if not video.exists():
        print(f"Error: {video} does not exist.")
        sys.exit(1)

    print(f"--- Antigravity Agent ---")
    print(f"Video: {video.name}")
    print(f"Task: {args.task}")
    print(f"SAM3: {settings.enable_sam3_tracking}")
    print(f"IndicASR: {settings.use_indic_asr}")
    print(f"Manipulation: {settings.manipulation_backend}")
    print()

    agent = MediaPipelineAgent()
    agent.initialize(str(video), args.task)
    
    result = agent.run()
    
    print()
    print("--- Result ---")
    print(f"Transcript: {result['transcript_length']} chars")
    print(f"Masks: {result['masks_count']}")
    
    if result['output_path']:
        print(f"Output: {result['output_path']}")
        
    if result['errors']:
        print(f"Errors: {result['errors']}")


if __name__ == "__main__":
    main()
