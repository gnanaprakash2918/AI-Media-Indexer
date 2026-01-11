"""Media conversion and optimization tools.

Usage:
    python -m tools.convert path/to/video.webm
    python -m tools.convert path/to/directory --workers 4
"""

import argparse
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path


def get_file_size_mb(path: Path) -> float:
    """Get file size in MB."""
    return path.stat().st_size / (1024 * 1024)


def get_video_duration(path: Path) -> float | None:
    """Get video duration in seconds using ffprobe."""
    try:
        cmd = [
            "ffprobe", "-v", "error",
            "-show_entries", "format=duration",
            "-of", "default=noprint_wrappers=1:nokey=1",
            str(path)
        ]
        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode == 0:
            return float(result.stdout.strip())
    except Exception:
        pass
    return None


def recommend_preset(file_size_mb: float, duration_seconds: float | None) -> str:
    """Recommend a preset based on file size and duration.
    
    Returns:
        Recommended preset name
    """
    # Estimate based on file size if duration not available
    if duration_seconds is None:
        # Rough estimate: assume ~5MB/min for typical video
        duration_seconds = (file_size_mb / 5) * 60

    duration_hours = duration_seconds / 3600

    if duration_hours >= 2:
        # Long videos (2+ hours) - prioritize speed
        return "ultrafast"
    elif duration_hours >= 0.5:
        # Medium videos (30min - 2hr) - balance
        return "fast"
    else:
        # Short videos (<30min) - can afford better quality
        return "medium"


def format_duration(seconds: float) -> str:
    """Format duration to human readable."""
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = int(seconds % 60)

    if hours > 0:
        return f"{hours}h {minutes}m {secs}s"
    elif minutes > 0:
        return f"{minutes}m {secs}s"
    else:
        return f"{secs}s"


def estimate_conversion_time(duration_seconds: float, preset: str) -> str:
    """Estimate conversion time based on preset."""
    # Rough multipliers (conversion time as fraction of video duration)
    multipliers = {
        "ultrafast": 0.3,  # ~30% of video duration
        "fast": 0.5,       # ~50% of video duration
        "medium": 1.0,     # ~100% of video duration
        "slow": 2.0,       # ~200% of video duration
    }
    mult = multipliers.get(preset, 0.5)
    est_seconds = duration_seconds * mult
    return format_duration(est_seconds)


def prompt_preset_choice(file_path: Path, recommended: str) -> str:
    """Interactive prompt to choose preset."""
    file_size_mb = get_file_size_mb(file_path)
    duration = get_video_duration(file_path)
    duration_str = format_duration(duration) if duration else "Unknown"

    print(f"\n{'='*60}")
    print(f"  File: {file_path.name}")
    print(f"  Size: {file_size_mb:.1f} MB | Duration: {duration_str}")
    print(f"{'='*60}")
    print()
    print("Choose encoding preset:")
    print()

    presets = [
        ("1", "ultrafast", "Fastest encoding, larger file size",
         estimate_conversion_time(duration or 600, "ultrafast") if duration else "~30% of video length"),
        ("2", "fast", "Good balance of speed and quality (RECOMMENDED for most)",
         estimate_conversion_time(duration or 600, "fast") if duration else "~50% of video length"),
        ("3", "medium", "Better quality, slower encoding",
         estimate_conversion_time(duration or 600, "medium") if duration else "~100% of video length"),
        ("4", "slow", "Best quality, slowest encoding (not recommended for long videos)",
         estimate_conversion_time(duration or 600, "slow") if duration else "~200% of video length"),
    ]

    for num, name, desc, est_time in presets:
        marker = " ★" if name == recommended else ""
        print(f"  [{num}] {name:12}{marker}")
        print(f"      {desc}")
        print(f"      Estimated time: {est_time}")
        print()

    print(f"  Recommended for this file: {recommended}")
    print()

    try:
        choice = input(f"Enter choice [1-4] or press Enter for '{recommended}': ").strip()
    except (EOFError, KeyboardInterrupt):
        print("\nUsing default preset.")
        return recommended

    preset_map = {"1": "ultrafast", "2": "fast", "3": "medium", "4": "slow"}

    if not choice:
        return recommended
    elif choice in preset_map:
        return preset_map[choice]
    else:
        print(f"Invalid choice. Using '{recommended}'.")
        return recommended


def convert_to_mp4(input_path: Path, output_path: Path | None = None, preset: str = "fast") -> bool:
    """Convert a video file to MP4 with faststart for instant seeking.
    
    Args:
        input_path: Path to input video file
        output_path: Output path (default: same name with .mp4 extension)
        preset: FFmpeg preset (ultrafast, fast, medium, slow)
    
    Returns:
        True if conversion succeeded
    """
    if output_path is None:
        output_path = input_path.with_suffix('.mp4')

    if output_path.exists():
        print(f"  ⏭ Skipping {input_path.name} - output already exists")
        return True

    cmd = [
        "ffmpeg", "-y",
        "-i", str(input_path),
        "-c:v", "libx264",
        "-preset", preset,
        "-crf", "23",
        "-c:a", "aac",
        "-b:a", "128k",
        "-movflags", "+faststart",
        "-progress", "pipe:1",
        str(output_path)
    ]

    print(f"  🔄 Converting: {input_path.name}")
    print(f"     Preset: {preset}")

    try:
        result = subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE, timeout=7200)  # 2hr timeout

        if result.returncode == 0 and output_path.exists():
            out_size = get_file_size_mb(output_path)
            print(f"  ✓ Done: {output_path.name} ({out_size:.1f} MB)")
            return True
        else:
            print(f"  ✗ Failed: {input_path.name}")
            if output_path.exists():
                output_path.unlink()  # Clean up failed output
            return False
    except subprocess.TimeoutExpired:
        print(f"  ✗ Timeout: {input_path.name} (exceeded 2 hours)")
        if output_path.exists():
            output_path.unlink()
        return False


def batch_convert(directory: Path, extensions: list[str] = [".webm", ".mkv", ".avi"],
                  workers: int = 2, preset: str = "fast", interactive: bool = True) -> tuple[int, int]:
    """Convert all videos in directory to MP4.
    
    Args:
        directory: Directory containing videos
        extensions: File extensions to convert
        workers: Number of parallel conversions
        preset: FFmpeg preset
        interactive: Whether to prompt for preset
    
    Returns:
        Tuple of (successful, failed) counts
    """
    files = []
    for ext in extensions:
        files.extend(directory.glob(f"*{ext}"))
        files.extend(directory.glob(f"*{ext.upper()}"))

    if not files:
        print(f"No files found with extensions {extensions}")
        return 0, 0

    # Calculate total size
    total_size_mb = sum(get_file_size_mb(f) for f in files)

    print(f"\n{'='*60}")
    print(f"  Found {len(files)} files to convert")
    print(f"  Total size: {total_size_mb:.1f} MB")
    print(f"  Workers: {workers}")
    print(f"{'='*60}\n")

    if interactive and len(files) > 0:
        # Recommend based on largest file
        largest = max(files, key=lambda f: f.stat().st_size)
        duration = get_video_duration(largest)
        recommended = recommend_preset(get_file_size_mb(largest), duration)
        preset = prompt_preset_choice(largest, recommended)
        print(f"\nUsing preset '{preset}' for all files.\n")

    success = 0
    failed = 0

    if workers == 1:
        for f in files:
            if convert_to_mp4(f, None, preset):
                success += 1
            else:
                failed += 1
    else:
        with ThreadPoolExecutor(max_workers=workers) as executor:
            futures = {executor.submit(convert_to_mp4, f, None, preset): f for f in files}

            for future in as_completed(futures):
                if future.result():
                    success += 1
                else:
                    failed += 1

    return success, failed


def main():
    parser = argparse.ArgumentParser(
        description="Convert videos to MP4 with faststart for instant seeking in browser",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
PRESETS:
  ultrafast  Fastest encoding, ~30%% of video duration, larger file
  fast       Good balance, ~50%% of video duration (DEFAULT for long videos)
  medium     Better quality, ~100%% of video duration (DEFAULT for short videos)
  slow       Best quality, ~200%% of video duration (not recommended for long)

EXAMPLES:
  # Interactive mode - shows menu to choose preset
  python -m tools.convert "C:\\Videos\\movie.webm"
  
  # Auto mode - uses smart defaults based on video duration
  python -m tools.convert "C:\\Videos\\movie.webm" -y
  
  # Force specific preset (skips interactive menu)
  python -m tools.convert "C:\\Videos\\movie.webm" --preset ultrafast
  
  # Batch convert entire folder with 4 parallel workers
  python -m tools.convert "C:\\Downloads\\Videos" --workers 4
  
  # Batch convert with specific preset, no prompts
  python -m tools.convert "C:\\Videos" --preset fast --workers 4 -y
  
  # Convert only .webm files (skip .mkv, .avi)
  python -m tools.convert "C:\\Videos" --extensions .webm

WHY CONVERT?
  WebM files (VP9 codec) have poor seek performance in browsers.
  Converting to MP4 with -movflags +faststart enables instant seeking
  for video segment playback in the AI-Media-Indexer search results.
        """
    )
    parser.add_argument("path", help="File or directory to convert")
    parser.add_argument("--preset", choices=["ultrafast", "fast", "medium", "slow"],
                        help="Encoding preset: ultrafast|fast|medium|slow (skips menu)")
    parser.add_argument("--workers", "-w", type=int, default=2,
                        help="Parallel workers for batch conversion (default: 2)")
    parser.add_argument("--extensions", "-e", nargs="+", default=[".webm", ".mkv", ".avi"],
                        help="File extensions to convert (default: .webm .mkv .avi)")
    parser.add_argument("--no-interactive", "-y", action="store_true",
                        help="Skip interactive prompts, use smart defaults")

    args = parser.parse_args()
    path = Path(args.path)

    if not path.exists():
        print(f"Error: {path} does not exist")
        sys.exit(1)

    interactive = not args.no_interactive and args.preset is None

    if path.is_file():
        # Single file conversion
        if path.suffix.lower() == '.mp4':
            print(f"File is already MP4: {path.name}")
            sys.exit(0)

        if interactive:
            duration = get_video_duration(path)
            recommended = recommend_preset(get_file_size_mb(path), duration)
            preset = prompt_preset_choice(path, recommended)
        else:
            if args.preset:
                preset = args.preset
            else:
                duration = get_video_duration(path)
                preset = recommend_preset(get_file_size_mb(path), duration)
                print(f"Auto-selected preset: {preset}")

        print()
        success = convert_to_mp4(path, preset=preset)
        print()
        sys.exit(0 if success else 1)
    else:
        # Batch conversion
        success, failed = batch_convert(
            path, args.extensions, args.workers,
            args.preset or "fast", interactive
        )
        print(f"\n{'='*60}")
        print(f"  Completed: {success} succeeded, {failed} failed")
        print(f"{'='*60}\n")
        sys.exit(0 if failed == 0 else 1)


if __name__ == "__main__":
    main()
