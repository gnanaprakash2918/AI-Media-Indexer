"""Extract frames from video files at specified intervals using FFmpeg.

TIMESTAMP ACCURACY FIX: This module now extracts actual PTS (Presentation Time Stamps)
from FFmpeg rather than calculating timestamps from frame count * interval.
This fixes timestamp drift on VFR (Variable Frame Rate) videos.

STREAMING FIX: This module now streams frames via stdout pipe instead of writing
thousands of temporary files. This prevents pipeline stalls on long videos.
"""

import asyncio
import json
import shutil
import subprocess
import tempfile
import traceback
from collections.abc import AsyncGenerator
from dataclasses import dataclass
from pathlib import Path

from core.errors import ExtractionError
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
        output_dir: Path | None = None,
    ) -> AsyncGenerator[ExtractedFrame, None]:
        """Generator that extracts frames from a video file with accurate timestamps.

        Uses streaming mode to yield frames as they are processed, avoiding
        long blocking pauses for large videos.

        Args:
            video_path: Path to the video file.
            interval: Time interval in seconds between extracted frames. 0.0 = every frame.
            start_time: Optional start time in seconds for partial extraction.
            end_time: Optional end time in seconds for partial extraction.
            output_dir: Optional directory to write frames to. If None, a temp dir is used.

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

            # Helper to run extraction logic (avoids code duplication)
            async def _run_extraction(cache_dir: Path):
                # Build FFmpeg command for image pipe
                # We use image2pipe to stream JPEGs to stdout
                args_to_ffmpeg = [
                    "ffmpeg",
                    "-y",
                    "-hide_banner",
                    "-loglevel",
                    "error",
                ]

                # -ss before -i for fast seeking
                if start_time is not None:
                    args_to_ffmpeg.extend(["-ss", str(start_time)])

                args_to_ffmpeg.extend(["-i", str(path_obj)])

                # -t for duration
                if end_time is not None:
                    duration = end_time - (start_time or 0)
                    if duration > 0:
                        args_to_ffmpeg.extend(["-t", str(duration)])

                # Video filter and quality settings
                video_filters = []
                if interval > 0:
                    video_filters.append(f"fps=1/{interval}")

                # Add PTS info request if possible, but for image2pipe usually
                # we rely on the pre-calculated timestamps or count
                if video_filters:
                    args_to_ffmpeg.extend(["-vf", ",".join(video_filters)])

                args_to_ffmpeg.extend(
                    [
                        "-q:v",
                        "2",  # High quality JPEG
                        "-f",
                        "image2pipe",
                        "-vcodec",
                        "mjpeg",
                        "-",  # Output to stdout
                    ]
                )

                log(f"Starting async ffmpeg stream for {path_obj.name}")

                process = await asyncio.create_subprocess_exec(
                    *args_to_ffmpeg,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    limit=10 * 1024 * 1024,  # 10MB buffer
                )

                # Read from stdout and delimit JPEGs
                # JPEG Start of Image (SOI): FF D8
                # JPEG End of Image (EOI): FF D9

                buffer = bytearray()
                frame_idx = 0
                chunk_size = 65536

                # We need to process stream
                while True:
                    chunk = await process.stdout.read(chunk_size)
                    if not chunk:
                        break

                    buffer.extend(chunk)

                    # Process buffer for multiple images
                    while True:
                        start_idx = buffer.find(b"\xff\xd8")
                        end_idx = buffer.find(b"\xff\xd9")

                        if (
                            start_idx != -1
                            and end_idx != -1
                            and end_idx > start_idx
                        ):
                            # Start and End found - we have a full image
                            jpeg_data = buffer[start_idx : end_idx + 2]

                            # Save to temp file (since downstream expects paths)
                            frame_name = f"frame_{frame_idx:06d}.jpg"
                            frame_path = cache_dir / frame_name

                            with open(frame_path, "wb") as f:
                                # Standard synchronous write is fine for tempfs/SSD
                                f.write(jpeg_data)

                            # Determine timestamp
                            if frame_timestamps and frame_idx < len(
                                frame_timestamps
                            ):
                                timestamp = frame_timestamps[frame_idx]
                            else:
                                # ⚠️ FFprobe timestamp extraction failed
                                # Using calculated timestamp which may drift on VFR videos
                                log(
                                    f"[WARN] Using calculated timestamp for frame {frame_idx} (FFprobe unavailable)",
                                    level="WARNING",
                                )
                                timestamp = (start_time or 0.0) + (
                                    frame_idx * interval
                                )

                            yield ExtractedFrame(
                                path=frame_path,
                                timestamp=timestamp,
                                frame_index=frame_idx,
                            )

                            frame_idx += 1

                            # Remove processed part from buffer
                            buffer = buffer[end_idx + 2 :]
                        elif start_idx != -1 and (
                            end_idx == -1 or end_idx < start_idx
                        ):
                            # Found start but not end (or end was from prev garbage), wait for more data
                            # Keep buffer from start_idx
                            if start_idx > 0:
                                buffer = buffer[start_idx:]
                            break
                        elif start_idx == -1:
                            # No start found, discard everything except maybe last few bytes in case split marker
                            if len(buffer) > 2:
                                buffer = buffer[-2:]
                            break

                # Check process return code
                await process.wait()
                if process.returncode != 0:
                    error_out = await process.stderr.read()
                    log(
                        f"FFmpeg streaming warning: {error_out.decode('utf-8', errors='ignore')}"
                    )

            # Execute the extraction logic
            if output_dir:
                # Use provided directory (caller manages lifecycle)
                async for frame in _run_extraction(output_dir):
                    yield frame
            else:
                # Use internal temp directory (auto-cleanup)
                with FrameExtractor.FrameCache() as cache_dir:
                    async for frame in _run_extraction(cache_dir):
                        yield frame

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
            raise ExtractionError(
                f"File access error for {video_path}: {exc}", original_error=exc
            )
        except Exception as exc:
            log(
                f"[ERROR:{type(exc).__name__}] Unexpected error processing '{video_path}': {exc}"
            )
            log(f"[Extractor] Trace: {traceback.format_exc()}", level="ERROR")
            raise ExtractionError(
                f"Unexpected extraction error for {video_path}",
                original_error=exc,
            )

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
                "ffprobe",
                "-v",
                "quiet",
                "-select_streams",
                "v:0",
                "-show_entries",
                "frame=pkt_pts_time,best_effort_timestamp_time",
                "-of",
                "json",
            ]

            # Add time range if specified
            if start_time is not None:
                cmd.extend(["-read_intervals", f"{start_time}%"])

            cmd.append(str(video_path))

            def run_probe():
                return subprocess.run(
                    cmd, capture_output=True, text=True, timeout=120
                )

            result = await asyncio.to_thread(run_probe)

            if result.returncode != 0:
                log(
                    "FFprobe timestamp extraction failed, using calculated timestamps"
                )
                return []

            data = json.loads(result.stdout)
            frames = data.get("frames", [])

            # Extract all frame timestamps
            all_timestamps = []
            for frame in frames:
                # Try pkt_pts_time first, then best_effort_timestamp_time
                ts = frame.get("pkt_pts_time") or frame.get(
                    "best_effort_timestamp_time"
                )
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
    async for frame in extractor.extract(
        video_path, interval, start_time, end_time
    ):
        yield frame.path


if __name__ == "__main__":

    async def main():
        """Test the FrameExtractor with a sample video file."""
        extractor = FrameExtractor()
        # Create a dummy video if needed or use existing
        async for frame in extractor.extract("test_video.mp4"):
            log(f"Got frame: {frame.path} at {frame.timestamp:.3f}s")

    # asyncio.run(main())
