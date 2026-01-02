"""Extract frames from video files at specified intervals using FFmpeg."""

import asyncio
import shutil
import subprocess
import tempfile
import traceback
from collections.abc import AsyncGenerator
from pathlib import Path

from core.utils.logger import log
from core.utils.observe import observe


class FrameExtractor:
    """Class to extract frames from video files at specified intervals."""

    class FrameCache:
        """Dont clutter the user's video folder with thousands of JPEGs."""

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
    ) -> AsyncGenerator[Path, None]:
        """Generator that extracts frames from a video file at specified intervals.

        Args:
            video_path: Path to the video file.
            interval: Time interval in seconds between extracted frames. 0.0 = every frame.
            start_time: Optional start time in seconds for partial extraction.
            end_time: Optional end time in seconds for partial extraction.
        
        Returns:
            An async generator yielding Paths to the extracted frame images.
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

            with FrameExtractor.FrameCache() as cache_dir:
                output_pattern = cache_dir / "frame_%04d.jpg"

                # Build FFmpeg command with optional time range
                args_to_ffmpeg = ["ffmpeg"]
                
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
                
                args_to_ffmpeg.extend([
                    "-q:v", "2",  # High quality JPEG
                    "-f", "image2",
                    str(output_pattern),
                ])

                log(f"Starting async ffmpeg process for {path_obj.name}")

                def run_ffmpeg():
                    return subprocess.run(
                        args_to_ffmpeg,
                        stdout=subprocess.PIPE,
                        stderr=subprocess.PIPE,
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
                for frame_path in frames:
                    yield frame_path

        except (
            ValueError,
            NotADirectoryError,
            FileNotFoundError,
            PermissionError,
            IsADirectoryError,
            OSError,
        ) as exc:
            log(f"[ERROR:{type(exc).__name__}] Cannot read '{video_path}': {exc}")
            return

        except Exception as exc:
            log(
                f"[ERROR:{type(exc).__name__}] Unexpected error "
                f"processing '{video_path}': {exc}",
            )
            traceback.print_exc()
            return


if __name__ == "__main__":

    async def main():
        """Test the FrameExtractor with a sample video file."""
        extractor = FrameExtractor()
        async for frame in extractor.extract("test_video.mp4"):
            log(f"Got frame: {frame}")

    # asyncio.run(main())
