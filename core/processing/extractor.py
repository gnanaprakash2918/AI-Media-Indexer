import shutil
from collections.abc import Generator
from pathlib import Path
import subprocess
from typing import Any

class FrameExtractor:
    class FrameCache:
        """Dont clutter the user's video folder with thousands of JPEGs."""
        def __init__(self, base: str | Path = ".", folder_name: str = ".frame_cache"):
            self.base = Path(base).resolve()
            self.path = self.base / folder_name

            # Fresh directory
            if self.path.exists():
                shutil.rmtree(self.path, ignore_errors=True)
            self.path.mkdir(parents=True, exist_ok=True)

            self._active = True

        def cleanup(self):
            """Manually delete the cache."""
            if self._active and self.path.exists():
                shutil.rmtree(self.path, ignore_errors=True)

            self._active = False

        # This is to enable context within 'with'
        def __enter__(self) -> Path:
            return self.path

        def __exit__(self, exc_type, exc, tb):
            self.cleanup()

    def extract(
        video_path: str | Path, interval: int = 2
    ) -> Generator[Path, None, None]:
        """
        Convert a video file into sequence of frames at regular intervals.
        interval=2 means "Take a screenshot every 2 seconds."

        Yields:
            Path to each extracted .jpg frame so the AI can process it immediately.
        """

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
                raise IsADirectoryError(f"Expected a file, but got a directory: {path_obj}")

            if not path_obj.is_file():
                raise FileNotFoundError(f"Path is not a file: {path_obj}")

            # Extract frames into temporary cache directory
            with FrameExtractor.FrameCache(base=path_obj.parent, folder_name=".frame_cache") as cache_dir:
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

                process = subprocess.Popen(
                    args=args_to_ffmpeg,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    text=True,
                    shell=False,
                )

                out, err = process.communicate()
                return_code = process.returncode

                if return_code != 0:
                    raise RuntimeError(
                        f"ffmpeg failed with code {return_code}: {err.strip() if err else ''}"
                    )

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
                f"Cannot read '{video_path}': {exc}"
            )
            return


if __name__ == "__main__":
    frame_ext = FrameExtractor().FrameCache()
    with frame_ext as cache:
        import time

        time.sleep(3)
        print(cache)

    print(frame_ext.path)

    # manual delete
    frame_ext.cleanup()
