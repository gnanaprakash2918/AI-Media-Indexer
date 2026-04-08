"""Media file utilities."""

from pathlib import Path


def has_audio_stream(path: Path) -> bool:
    """Check if media file has audio stream using ffprobe."""
    try:
        from core.processing.prober import get_probe_sync

        data = get_probe_sync(path)
        return any(
            s.get("codec_type") == "audio" for s in data.get("streams", [])
        )
    except Exception:
        return False


def get_duration(path: Path) -> float:
    """Get media duration in seconds using ffprobe."""
    try:
        from core.processing.prober import get_probe_sync

        data = get_probe_sync(path)
        return float(data.get("format", {}).get("duration", 0.0))
    except Exception:
        return 0.0


def get_fps(path: Path) -> float:
    """Get video FPS using ffprobe."""
    try:
        from core.processing.prober import get_probe_sync

        data = get_probe_sync(path)
        for s in data.get("streams", []):
            if s.get("codec_type") == "video":
                fps_str = s.get("r_frame_rate", "0/0")
                if "/" in fps_str:
                    num, den = fps_str.split("/")
                    if float(den) > 0:
                        return float(num) / float(den)
                elif fps_str:
                    return float(fps_str)
    except Exception:
        pass
    return 30.0
