from __future__ import annotations
from pathlib import Path
from dataclasses import dataclass

from config import settings
from core.utils.logger import log


@dataclass
class SceneInfo:
    start_time: float
    end_time: float
    start_frame: int
    end_frame: int
    mid_frame: int
    mid_time: float


def detect_scenes(video_path: Path | str) -> list[SceneInfo]:
    try:
        from scenedetect import detect, AdaptiveDetector, FrameTimecode
    except ImportError:
        log("scenedetect not installed")
        return []
    
    path = Path(video_path)
    if not path.exists():
        return []
    
    try:
        scene_list = detect(
            str(path),
            AdaptiveDetector(
                adaptive_threshold=settings.scene_detect_threshold,
                min_scene_len=int(settings.scene_detect_min_length * 30),  # assuming 30fps
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
        
        scenes.append(SceneInfo(
            start_time=start_time,
            end_time=end_time,
            start_frame=start_frame,
            end_frame=end_frame,
            mid_frame=mid_frame,
            mid_time=mid_time,
        ))
    
    log(f"Detected {len(scenes)} scenes in {path.name}")
    return scenes


def extract_scene_frame(video_path: Path | str, timestamp: float) -> bytes | None:
    import subprocess
    path = Path(video_path)
    if not path.exists():
        return None
    
    cmd = [
        "ffmpeg", "-y",
        "-ss", str(timestamp),
        "-i", str(path),
        "-frames:v", "1",
        "-f", "image2pipe",
        "-vcodec", "mjpeg",
        "-q:v", "5",
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
