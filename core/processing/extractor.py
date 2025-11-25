import shutil
import subprocess
import tempfile
from collections.abc import Generator
from pathlib import Path


class FrameExtractor:
    class FrameCache:
        """Dont clutter the user's video folder with thousands of JPEGs."""

        def __init__(self):
            self.path = Path(tempfile.mkdtemp(prefix="media_agent_frames_"))
            self._active = True

        def cleanup(self):
            """Manually delete the cache."""
            if self._active and self.path.exists():
                shutil.rmtree(self.path, ignore_errors=True)

            self._active = False

        # This is to enable context within 'with'
        def __enter__(self):
            return self.path

        def __exit__(self, exc_type, exc, tb):
            self.cleanup()

    def extract(
        self, video_path: str | Path, interval: int = 2
    ) -> Generator[Path, None, None]:
        # Convert a video file into sequence of frames at regular intervals
        # interval=2 means "Take a screenshot every 2 seconds."

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

            # Extract frames into temporary cache directory
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

                process = subprocess.run(
                    args_to_ffmpeg,
                    capture_output=True,
                    text=True,
                    shell=False,
                    check=False,
                )

                if process.returncode != 0:
                    print(
                        f"[ERROR:FFmpeg] Failed to extract frames from '{video_path}': "
                        f"{process.stderr.strip() if process.stderr else ''}"
                    )
                    return

                for frame_path in sorted(cache_dir.glob("frame_*.jpg")):
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
                f"[ERROR:{type(exc).__name__}] "
                f"Cannot read '{video_path}': {exc}"
            )
            return

        except Exception as exc:  # pylint: disable=broad-except
            print(
                f"[ERROR:{type(exc).__name__}] "
                f"Unexpected error for '{video_path}': {exc}"
            )
            return


if __name__ == "__main__":
    frame_ext = FrameExtractor.FrameCache()
    print(frame_ext.path)
    frame_ext.cleanup()
