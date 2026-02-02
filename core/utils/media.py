"""Media file utilities."""

import subprocess
from pathlib import Path


def has_audio_stream(path: Path) -> bool:
    """Check if media file has audio stream using ffprobe."""
    try:
        cmd = [
            "ffprobe",
            "-v",
            "error",
            "-select_streams",
            "a",
            "-show_entries",
            "stream=codec_type",
            "-of",
            "csv=p=0",
            str(path),
        ]
        output = (
            subprocess.check_output(cmd, stderr=subprocess.DEVNULL)
            .decode()
            .strip()
        )
        return bool(output)
    except Exception:
        return False


def get_duration(path: Path) -> float:
    """Get media duration in seconds using ffprobe."""
    try:
        cmd = [
            "ffprobe",
            "-v",
            "error",
            "-show_entries",
            "format=duration",
            "-of",
            "default=noprint_wrappers=1:nokey=1",
            str(path),
        ]
        output = (
            subprocess.check_output(cmd, stderr=subprocess.DEVNULL)
            .decode()
            .strip()
        )
        return float(output)
    except Exception:
        return 0.0
