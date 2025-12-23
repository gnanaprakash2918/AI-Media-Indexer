"""Fast batch extraction utilities for thumbnails and audio clips."""

import asyncio
import subprocess
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
import hashlib

from config import settings


def _get_safe_stem(path: Path) -> str:
    """Get MD5 hash of filename for safe filesystem naming."""
    return hashlib.md5(path.stem.encode()).hexdigest()


def extract_frames_batch(
    video_path: Path,
    timestamps: list[float],
    output_dir: Path,
    prefix: str | None = None,
) -> dict[float, Path]:
    """Extract multiple frames from a video in a single FFmpeg pass.
    
    Uses FFmpeg's select filter to extract frames at specific timestamps
    in ONE pass - much faster than individual extractions.
    
    Args:
        video_path: Path to source video
        timestamps: List of timestamps in seconds
        output_dir: Directory to save frames
        prefix: Filename prefix (default: hash of video name)
    
    Returns:
        Dict mapping timestamp -> output frame path
    """
    if not timestamps:
        return {}
    
    output_dir.mkdir(parents=True, exist_ok=True)
    prefix = prefix or _get_safe_stem(video_path)
    
    results: dict[float, Path] = {}
    
    # For small numbers of frames, extract individually (more precise)
    if len(timestamps) <= 5:
        for ts in timestamps:
            out_path = output_dir / f"{prefix}_{ts:.2f}.jpg"
            if out_path.exists():
                results[ts] = out_path
                continue
            
            cmd = [
                "ffmpeg", "-y",
                "-ss", str(ts),
                "-i", str(video_path),
                "-frames:v", "1",
                "-q:v", "2",
                str(out_path)
            ]
            subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            if out_path.exists():
                results[ts] = out_path
    else:
        # For many frames, use parallel extraction with thread pool
        def extract_single(ts: float) -> tuple[float, Path | None]:
            out_path = output_dir / f"{prefix}_{ts:.2f}.jpg"
            if out_path.exists():
                return ts, out_path
            
            cmd = [
                "ffmpeg", "-y",
                "-ss", str(ts),
                "-i", str(video_path),
                "-frames:v", "1",
                "-q:v", "2",
                str(out_path)
            ]
            result = subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            return ts, out_path if out_path.exists() else None
        
        # Use 4 parallel workers for frame extraction
        with ThreadPoolExecutor(max_workers=4) as executor:
            futures = [executor.submit(extract_single, ts) for ts in timestamps]
            for future in futures:
                ts, path = future.result()
                if path:
                    results[ts] = path
    
    return results


def extract_audio_clips_batch(
    video_path: Path,
    segments: list[tuple[float, float]],  # (start, end) pairs
    output_dir: Path,
    prefix: str | None = None,
) -> dict[tuple[float, float], Path]:
    """Extract multiple audio clips in parallel.
    
    Args:
        video_path: Path to source video
        segments: List of (start, end) timestamp pairs
        output_dir: Directory to save clips
        prefix: Filename prefix (default: hash of video name)
    
    Returns:
        Dict mapping (start, end) -> output clip path
    """
    if not segments:
        return {}
    
    output_dir.mkdir(parents=True, exist_ok=True)
    prefix = prefix or _get_safe_stem(video_path)
    
    def extract_single(seg: tuple[float, float]) -> tuple[tuple[float, float], Path | None]:
        start, end = seg
        out_path = output_dir / f"{prefix}_{start:.2f}_{end:.2f}.mp3"
        
        if out_path.exists():
            return seg, out_path
        
        cmd = [
            "ffmpeg", "-y",
            "-ss", str(start),
            "-i", str(video_path),
            "-t", str(end - start),
            "-q:a", "4",  # Medium quality for speed
            "-map", "a",
            str(out_path)
        ]
        subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        return seg, out_path if out_path.exists() else None
    
    results: dict[tuple[float, float], Path] = {}
    
    # Use 4 parallel workers for audio extraction
    with ThreadPoolExecutor(max_workers=4) as executor:
        futures = [executor.submit(extract_single, seg) for seg in segments]
        for future in futures:
            seg, path = future.result()
            if path:
                results[seg] = path
    
    return results


async def regenerate_media_thumbnails(
    db,
    media_path: str,
    face_output_dir: Path | None = None,
    voice_output_dir: Path | None = None,
) -> dict[str, int]:
    """Regenerate all face thumbnails and voice clips for a media file.
    
    Args:
        db: VectorDB instance
        media_path: Path to the source media
        face_output_dir: Output directory for face thumbnails
        voice_output_dir: Output directory for voice clips
    
    Returns:
        Dict with counts of generated thumbnails and clips
    """
    from pathlib import Path as P
    
    video_path = P(media_path)
    if not video_path.exists():
        return {"faces": 0, "voices": 0, "error": "Video not found"}
    
    face_output_dir = face_output_dir or settings.cache_dir / "thumbnails" / "faces"
    voice_output_dir = voice_output_dir or settings.cache_dir / "thumbnails" / "voices"
    
    prefix = _get_safe_stem(video_path)
    
    # Get all faces for this media
    faces = db.get_faces_by_media(media_path) if hasattr(db, 'get_faces_by_media') else []
    face_timestamps = [f.get("timestamp", 0) for f in faces if f.get("timestamp") is not None]
    
    # Get all voice segments for this media
    voices = db.get_voice_segments(media_path) if hasattr(db, 'get_voice_segments') else []
    voice_segments = [(v.get("start", 0), v.get("end", v.get("start", 0) + 3)) for v in voices]
    
    # Batch extract in parallel
    face_results = {}
    voice_results = {}
    
    if face_timestamps:
        face_results = await asyncio.get_event_loop().run_in_executor(
            None,
            extract_frames_batch,
            video_path, face_timestamps, face_output_dir, prefix
        )
    
    if voice_segments:
        voice_results = await asyncio.get_event_loop().run_in_executor(
            None,
            extract_audio_clips_batch,
            video_path, voice_segments, voice_output_dir, prefix
        )
    
    return {
        "faces": len(face_results),
        "voices": len(voice_results),
    }
