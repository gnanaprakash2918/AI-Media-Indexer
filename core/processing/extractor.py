from pathlib import Path
from collections.abc import Generator

class FrameExtractor:
    def extract(video_path: str | Path, interval: int = 2) -> Generator[Path, None, None]:
        # Convert a video file into sequence of frames at regular intervals
        if isinstance(video_path, str):
            if video_path.strip() == '':
                return

            path_obj = Path(video_path)
        else:
            path_obj = video_path
