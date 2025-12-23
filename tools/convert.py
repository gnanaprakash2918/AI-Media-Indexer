"""Media conversion and optimization tools."""

import argparse
import subprocess
import sys
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed


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
        print(f"  Skipping {input_path.name} - output already exists")
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
        str(output_path)
    ]
    
    print(f"  Converting: {input_path.name} -> {output_path.name}")
    result = subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)
    
    if result.returncode == 0:
        print(f"  ✓ Done: {output_path.name}")
        return True
    else:
        print(f"  ✗ Failed: {input_path.name}")
        return False


def batch_convert(directory: Path, extensions: list[str] = [".webm", ".mkv", ".avi"], 
                  workers: int = 2, preset: str = "fast") -> tuple[int, int]:
    """Convert all videos in directory to MP4.
    
    Args:
        directory: Directory containing videos
        extensions: File extensions to convert
        workers: Number of parallel conversions
        preset: FFmpeg preset
    
    Returns:
        Tuple of (successful, failed) counts
    """
    files = []
    for ext in extensions:
        files.extend(directory.glob(f"*{ext}"))
    
    if not files:
        print(f"No files found with extensions {extensions}")
        return 0, 0
    
    print(f"Found {len(files)} files to convert")
    
    success = 0
    failed = 0
    
    with ThreadPoolExecutor(max_workers=workers) as executor:
        futures = {executor.submit(convert_to_mp4, f, None, preset): f for f in files}
        
        for future in as_completed(futures):
            if future.result():
                success += 1
            else:
                failed += 1
    
    return success, failed


def main():
    parser = argparse.ArgumentParser(description="Convert videos to MP4 with faststart")
    parser.add_argument("path", help="File or directory to convert")
    parser.add_argument("--preset", default="fast", choices=["ultrafast", "fast", "medium", "slow"],
                        help="FFmpeg preset (default: fast)")
    parser.add_argument("--workers", type=int, default=2, help="Parallel workers for batch (default: 2)")
    parser.add_argument("--extensions", nargs="+", default=[".webm", ".mkv", ".avi"],
                        help="Extensions to convert (default: .webm .mkv .avi)")
    
    args = parser.parse_args()
    path = Path(args.path)
    
    if not path.exists():
        print(f"Error: {path} does not exist")
        sys.exit(1)
    
    if path.is_file():
        success = convert_to_mp4(path, preset=args.preset)
        sys.exit(0 if success else 1)
    else:
        success, failed = batch_convert(path, args.extensions, args.workers, args.preset)
        print(f"\nCompleted: {success} succeeded, {failed} failed")
        sys.exit(0 if failed == 0 else 1)


if __name__ == "__main__":
    main()
