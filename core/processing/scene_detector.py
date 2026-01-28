"""Video scene detection utilities using adaptive thresholding."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from config import settings
from core.utils.logger import log


@dataclass
class SceneInfo:
    """Contains timing and frame metadata for a detected video scene."""

    start_time: float
    end_time: float
    start_frame: int
    end_frame: int
    mid_frame: int
    mid_time: float


def _get_video_fps(video_path: Path) -> float:
    """Get actual video FPS using FFprobe.
    
    Args:
        video_path: Path to the video file.
        
    Returns:
        Frames per second (defaults to 30.0 if probe fails).
    """
    import subprocess
    
    try:
        cmd = [
            'ffprobe', '-v', 'quiet',
            '-select_streams', 'v:0',
            '-show_entries', 'stream=r_frame_rate',
            '-of', 'csv=p=0',
            str(video_path)
        ]
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
        
        if result.returncode == 0 and result.stdout.strip():
            # FFprobe returns FPS as fraction like "30000/1001" or "30/1"
            fps_str = result.stdout.strip()
            if '/' in fps_str:
                num, den = fps_str.split('/')
                fps = float(num) / float(den)
            else:
                fps = float(fps_str)
            
            log(f"Detected video FPS: {fps:.2f}")
            return fps
    except Exception as e:
        log(f"FPS detection failed, using 30fps fallback: {e}")
    
    return 30.0  # Fallback


async def detect_scenes(video_path: Path | str) -> list[SceneInfo]:
    """Detects logical scene boundaries in a video using adaptive thresholding.

    Args:
        video_path: Path to the input video file.

    Returns:
        A list of SceneInfo objects, one for each detected scene.
    """
    try:
        import scenedetect  # type: ignore

        adaptive_detector_cls = scenedetect.AdaptiveDetector
        detect = scenedetect.detect
    except ImportError:
        log("scenedetect not installed")
        return []

    path = Path(video_path)
    if not path.exists():
        return []

    # Get actual video FPS instead of assuming 30fps
    fps = _get_video_fps(path)

    import asyncio

    try:
        scene_list = await asyncio.to_thread(
            detect,
            str(path),
            adaptive_detector_cls(
                adaptive_threshold=settings.scene_detect_threshold,
                min_scene_len=int(
                    settings.scene_detect_min_length * fps
                ),
            ),
            show_progress=False,
        )
    except Exception as e:
        log(f"Scene detection failed: {e}")
        return []

    scenes = []
    for start_tc, end_tc in scene_list:
        start_time = start_tc.get_seconds()
        end_time = end_tc.get_seconds()
        start_frame = start_tc.get_frames()
        end_frame = end_tc.get_frames()
        mid_frame = (start_frame + end_frame) // 2
        mid_time = (start_time + end_time) / 2.0

        scenes.append(
            SceneInfo(
                start_time=start_time,
                end_time=end_time,
                start_frame=start_frame,
                end_frame=end_frame,
                mid_frame=mid_frame,
                mid_time=mid_time,
            )
        )

    log(f"Detected {len(scenes)} scenes in {path.name}")
    return scenes


def extract_scene_frame(
    video_path: Path | str, timestamp: float
) -> bytes | None:
    """Extracts a single frame from a video at a specific timestamp.

    Args:
        video_path: Path to the video file.
        timestamp: The time in seconds to extract the frame from.

    Returns:
        The JPEG-encoded frame as bytes, or None if extraction fails.
    """
    import subprocess

    path = Path(video_path)
    if not path.exists():
        return None

    cmd = [
        "ffmpeg",
        "-y",
        "-ss",
        str(timestamp),
        "-i",
        str(path),
        "-frames:v",
        "1",
        "-f",
        "image2pipe",
        "-vcodec",
        "mjpeg",
        "-q:v",
        "5",
        "pipe:1",
    ]
    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            timeout=10,
        )
        if result.returncode == 0 and result.stdout:
            return result.stdout
    except Exception:
        pass
    return None
