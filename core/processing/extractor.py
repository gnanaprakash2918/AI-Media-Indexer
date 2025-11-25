from pathlib import Path
from collections.abc import Generator

class FrameExtractor:
    def extract(video_path: str | Path, interval: int = 2) -> Generator[Path, None, None]:
        # Convert a video file into sequence of frames at regular intervals
        # interval=2 means "Take a screenshot every 2 seconds."

        try:
            if isinstance(video_path, str):
                if video_path.strip() == '':
                    raise ValueError("Provided path is empty or whitespace.")

                path_obj = Path(video_path)
            else:
                path_obj = video_path

            if not path_obj.exists():
                raise FileNotFoundError(
                    f"Path does not exist: {video_path}"
                )

            if not path_obj.is_dir():
                raise IsADirectoryError(
                    f"Path is not a directory: {path_obj}"
                )
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
