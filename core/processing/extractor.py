import asyncio
import shutil
import tempfile
from collections.abc import AsyncGenerator
from pathlib import Path


class FrameExtractor:
    class FrameCache:
        """Dont clutter the user's video folder with thousands of JPEGs."""

        def __init__(self):
            self.path = Path(tempfile.mkdtemp(prefix="media_agent_frames_"))
            self._active = True

        def cleanup(self):
            if self._active and self.path.exists():
                shutil.rmtree(self.path, ignore_errors=True)
            self._active = False

        def __enter__(self):
            return self.path

        def __exit__(self, exc_type, exc, tb):
            self.cleanup()

    async def extract(
        self, video_path: str | Path, interval: int = 2
    ) -> AsyncGenerator[Path, None]:
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
                    f"Expected a file, but got a directory: {path_obj}"
                )

            if not path_obj.is_file():
                raise FileNotFoundError(f"Path is not a file: {path_obj}")

            with FrameExtractor.FrameCache() as cache_dir:
                output_pattern = cache_dir / "frame_%04d.jpg"

                args_to_ffmpeg = [
                    "ffmpeg",
                    "-i",
                    str(path_obj),
                    "-vf",
                    f"fps=1/{interval}",
                    "-q:v",
                    "2",
                    "-f",
                    "image2",
                    str(output_pattern),
                ]

                print(f"Starting async ffmpeg process for {path_obj.name}")
                process = await asyncio.create_subprocess_exec(
                    *args_to_ffmpeg,
                    stdout=asyncio.subprocess.PIPE,
                    stderr=asyncio.subprocess.PIPE,
                )

                _, stderr = await process.communicate()

                if process.returncode != 0:
                    print(
                        f"[ERROR:FFmpeg] Failed to extract frames from '{video_path}': "
                        f"{stderr.decode().strip() if stderr else ''}"
                    )
                    return

                # Sort and yield
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
            print(
                f"[ERROR:{type(exc).__name__}] Cannot read '{video_path}': {exc}"
            )
            return

        except Exception as exc:
            print(
                f"[ERROR:{type(exc).__name__}] Unexpected error processing '{video_path}': {exc}"
            )
            return


if __name__ == "__main__":

    async def main():
        extractor = FrameExtractor()
        async for frame in extractor.extract("test_video.mp4"):
            print(f"Got frame: {frame}")

    # asyncio.run(main())
