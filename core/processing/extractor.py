"""Extract frames from video files at specified intervals using FFmpeg.

TIMESTAMP ACCURACY FIX: This module now extracts actual PTS (Presentation Time Stamps)
from FFmpeg rather than calculating timestamps from frame count * interval.
This fixes timestamp drift on VFR (Variable Frame Rate) videos.
"""

import asyncio
import json
import re
import shutil
import subprocess
import tempfile
import traceback
from collections.abc import AsyncGenerator
from dataclasses import dataclass
from pathlib import Path

from core.utils.logger import log
from core.utils.observe import observe


@dataclass
class ExtractedFrame:
    """A frame extracted from video with its actual timestamp.
    
    Attributes:
        path: Path to the extracted frame image file.
        timestamp: Actual presentation timestamp in seconds (from PTS, not calculated).
        frame_index: Sequential frame index (0-based).
    """
    path: Path
    timestamp: float
    frame_index: int


class FrameExtractor:
    """Class to extract frames from video files with accurate PTS timestamps.
    
    DESIGN DECISION: We extract actual timestamps from FFmpeg rather than
    calculating them as (frame_count * interval). This handles VFR videos
    correctly and prevents timestamp drift on long videos.
    """

    class FrameCache:
        """Don't clutter the user's video folder with thousands of JPEGs."""

        def __init__(self):
            """Initialize temporary directory for frame storage."""
            self.path = Path(tempfile.mkdtemp(prefix="media_agent_frames_"))
            self._active = True

        def cleanup(self):
            """Remove the temporary directory and its contents."""
            if self._active and self.path.exists():
                shutil.rmtree(self.path, ignore_errors=True)
            self._active = False

        def __enter__(self):
            """Enter the context manager, returning the path to the temp directory."""
            return self.path

        def __exit__(self, exc_type, exc, tb):
            """Exit the context manager, cleaning up the temp directory."""
            self.cleanup()

    @observe("frame_extract")
    async def extract(
        self,
        video_path: str | Path,
        interval: float = 2.0,
        start_time: float | None = None,
        end_time: float | None = None,
    ) -> AsyncGenerator[ExtractedFrame, None]:
        """Generator that extracts frames from a video file with accurate timestamps.

        Args:
            video_path: Path to the video file.
            interval: Time interval in seconds between extracted frames. 0.0 = every frame.
            start_time: Optional start time in seconds for partial extraction.
            end_time: Optional end time in seconds for partial extraction.

        Yields:
            ExtractedFrame objects containing path and actual PTS timestamp.
        """
        # Store time offset for timestamp calculation
        self._time_offset = start_time or 0.0

        # Non-blocking async generator for frame extraction
        try:
            if isinstance(video_path, str):
                if video_path.strip() == "":
                    raise ValueError("Provided path is empty or whitespace.")
                path_obj = Path(video_path)
            else:
                path_obj = video_path

            if not path_obj.exists():
                raise FileNotFoundError(f"Path does not exist: {path_obj}")

            if path_obj.is_dir():
                raise IsADirectoryError(
                    f"Expected a file, but got a directory: {path_obj}",
                )

            if not path_obj.is_file():
                raise FileNotFoundError(f"Path is not a file: {path_obj}")

            # Get frame timestamps using FFprobe first (for accuracy)
            frame_timestamps = await self._get_frame_timestamps(
                path_obj, interval, start_time, end_time
            )

            with FrameExtractor.FrameCache() as cache_dir:
                # Use timestamp-based filenames for accurate mapping
                output_pattern = cache_dir / "frame_%06d.jpg"

                # Build FFmpeg command with optional time range
                args_to_ffmpeg = ["ffmpeg", "-y", "-hide_banner", "-loglevel", "error"]

                # -ss before -i for fast seeking (input seeking)
                if start_time is not None:
                    args_to_ffmpeg.extend(["-ss", str(start_time)])

                args_to_ffmpeg.extend(["-i", str(path_obj)])

                # -t for duration (if end_time specified)
                if end_time is not None:
                    duration = end_time - (start_time or 0)
                    if duration > 0:
                        args_to_ffmpeg.extend(["-t", str(duration)])

                # Video filter and quality settings
                video_filters = []

                # Zero interval means "every frame" - skip fps filter
                if interval > 0:
                    # FPS = 1 / interval (e.g., 0.5s -> 2fps)
                    video_filters.append(f"fps=1/{interval}")

                if video_filters:
                    args_to_ffmpeg.extend(["-vf", ",".join(video_filters)])

                args_to_ffmpeg.extend(
                    [
                        "-q:v",
                        "2",  # High quality JPEG
                        "-f",
                        "image2",
                        "-start_number", "0",
                        str(output_pattern),
                    ]
                )

                log(f"Starting async ffmpeg process for {path_obj.name}")

                def run_ffmpeg():
                    return subprocess.run(
                        args_to_ffmpeg,
                        capture_output=True,
                        text=True,
                        encoding="utf-8",
                        errors="replace",
                    )

                proc = await asyncio.to_thread(run_ffmpeg)

                if proc.returncode != 0:
                    log(
                        "[ERROR:FFmpeg] Failed to extract frames from "
                        f"'{video_path}': {proc.stderr.strip() if proc.stderr else ''}",
                    )
                    return

                frames = sorted(cache_dir.glob("frame_*.jpg"))
                
                for idx, frame_path in enumerate(frames):
                    # Use actual PTS if available, otherwise calculate
                    if frame_timestamps and idx < len(frame_timestamps):
                        actual_ts = frame_timestamps[idx]
                    else:
                        # Fallback: calculated timestamp (less accurate for VFR)
                        actual_ts = (start_time or 0.0) + (idx * interval)
                    
                    yield ExtractedFrame(
                        path=frame_path,
                        timestamp=actual_ts,
                        frame_index=idx
                    )

        except (
            ValueError,
            NotADirectoryError,
            FileNotFoundError,
            PermissionError,
            IsADirectoryError,
            OSError,
        ) as exc:
            log(
                f"[ERROR:{type(exc).__name__}] Cannot read '{video_path}': {exc}"
            )
            return

        except Exception as exc:
            log(
                f"[ERROR:{type(exc).__name__}] Unexpected error "
                f"processing '{video_path}': {exc}",
            )
            traceback.print_exc()
            return

    async def _get_frame_timestamps(
        self,
        video_path: Path,
        interval: float,
        start_time: float | None,
        end_time: float | None,
    ) -> list[float]:
        """Extract actual PTS timestamps for sampled frames using FFprobe.
        
        This is the key fix for VFR (Variable Frame Rate) videos.
        
        Args:
            video_path: Path to the video file.
            interval: Sampling interval in seconds.
            start_time: Optional start time.
            end_time: Optional end time.
            
        Returns:
            List of actual timestamps in seconds.
        """
        try:
            # Build FFprobe command to get frame timestamps
            cmd = [
                'ffprobe', '-v', 'quiet',
                '-select_streams', 'v:0',
                '-show_entries', 'frame=pkt_pts_time,best_effort_timestamp_time',
                '-of', 'json',
            ]
            
            # Add time range if specified
            if start_time is not None:
                cmd.extend(['-read_intervals', f'{start_time}%'])
            
            cmd.append(str(video_path))
            
            def run_probe():
                return subprocess.run(
                    cmd, capture_output=True, text=True, timeout=120
                )
            
            result = await asyncio.to_thread(run_probe)
            
            if result.returncode != 0:
                log(f"FFprobe timestamp extraction failed, using calculated timestamps")
                return []
            
            data = json.loads(result.stdout)
            frames = data.get('frames', [])
            
            # Extract all frame timestamps
            all_timestamps = []
            for frame in frames:
                # Try pkt_pts_time first, then best_effort_timestamp_time
                ts = frame.get('pkt_pts_time') or frame.get('best_effort_timestamp_time')
                if ts is not None:
                    all_timestamps.append(float(ts))
            
            if not all_timestamps:
                return []
            
            # Sample at the specified interval
            if interval <= 0:
                return all_timestamps
            
            sampled_timestamps = []
            last_sampled = -interval  # Ensures first frame is sampled
            
            for ts in all_timestamps:
                if ts >= last_sampled + interval:
                    # Apply start/end time filters
                    if start_time and ts < start_time:
                        continue
                    if end_time and ts > end_time:
                        break
                    sampled_timestamps.append(ts)
                    last_sampled = ts
            
            log(f"Extracted {len(sampled_timestamps)} actual PTS timestamps")
            return sampled_timestamps
            
        except json.JSONDecodeError:
            log("Failed to parse FFprobe JSON output")
            return []
        except subprocess.TimeoutExpired:
            log("FFprobe timestamp extraction timed out")
            return []
        except Exception as e:
            log(f"Timestamp extraction failed: {e}")
            return []


# Legacy compatibility: yield just path for backward compatibility
async def extract_frames_legacy(
    video_path: str | Path,
    interval: float = 2.0,
    start_time: float | None = None,
    end_time: float | None = None,
) -> AsyncGenerator[Path, None]:
    """Legacy wrapper that yields only paths (for backward compatibility)."""
    extractor = FrameExtractor()
    async for frame in extractor.extract(video_path, interval, start_time, end_time):
        yield frame.path


if __name__ == "__main__":

    async def main():
        """Test the FrameExtractor with a sample video file."""
        extractor = FrameExtractor()
        async for frame in extractor.extract("test_video.mp4"):
            log(f"Got frame: {frame.path} at {frame.timestamp:.3f}s")

    # asyncio.run(main())

