import os
from pathlib import Path
from typing import Generator, Iterable
import magic
import datetime
from core.schemas import MediaType, MediaAsset

class LibraryScanner:
    """System to efficiently find media files (images, audio, video) under a directory."""

    DEFAULT_EXCLUDES: set[str] = {
        ".git", ".hg", ".svn",
        ".venv", "venv", ".env", "env"
        "__pycache__", ".mypy_cache", ".pytest_cache",
        "node_modules", ".cache", ".local",
        ".Trash", "lost+found",
        ".DS_Store", "__MACOSX",
        "Thumbs.db", "desktop.ini",
        "dist", "build", ".egg-info",
        ".idea", ".vscode",
    }

    VIDEO_EXTS: set[str] = {'.mp4', '.mkv', '.mov', '.avi', '.webm', '.m4v'}
    AUDIO_EXTS: set[str] = {'.mp3', '.wav', '.aac', '.flac', '.m4a', '.ogg'}
    IMAGE_EXTS: set[str] = {'.jpg', '.jpeg', '.png', '.webp', '.bmp', '.tiff'}

    def _get_media_type(self, extension: str):
        """Determines media type based on extension."""
        extension = extension.lower()
        if extension in self.VIDEO_EXTS:
            return MediaType.VIDEO
        elif extension in self.AUDIO_EXTS:
            return MediaType.AUDIO
        elif extension in self.IMAGE_EXTS:
            return MediaType.IMAGE

        return None


    def _is_media_file(self, file_path: Path) -> tuple[bool, str]:
        """
        Check whether a file is a media file (image, audio, or video).

        This function determines the MIME type of the file using the `magic` library
        and classifies it as one of: "image", "video", or "audio".

        Args:
            file_path: Path to the file being examined.

        Returns:
            A tuple (is_media, media_type):
                is_media: True if the file is recognized as a media file, False otherwise.
                media_type: One of "image", "video", "audio", or "none".
        """

        SMALL_FILE_SIZE_THRESHOLD: int = 4 * 1024

        try:
            file_stats = file_path.stat()
            file_size = file_stats.st_size

            if file_size > SMALL_FILE_SIZE_THRESHOLD:
                ext = file_path.suffix.lower()

            else:
                mime = magic.from_file(str(file_path), mime=True)

                for prefix, media_type_itr in (
                        ("image/", "image"),
                        ("video/", "video"),
                        ("audio/", "audio"),
                ):
                    if mime.startswith(prefix):
                        return True, media_type_itr

        except (FileNotFoundError, PermissionError, IsADirectoryError, OSError) as e:
            print(f"[ERROR:{type(e).__name__}] Cannot read '{file_path}': {e}")
            return False, "none"

        except Exception as e:
            print(f"[ERROR:{type(e).__name__}] Cannot read '{file_path}': {e}")
            return False, "none"

        return False, "none"

    def scan(self, root_path: str | Path, excluded_dirs: Iterable[str] | None = None) -> Generator[MediaAsset, None, None]:
        """
        Recursively yield media files (images, audio, video) under the given directory.

        Args:
            root_path: Root directory to scan. Can be a string or a Path object.
            excluded_dirs: Iterable of directory or file names to exclude by name. If None, a default set of common project/system directories is used.

        Yields:
            Tuples of (file_path, media_type), where:
                file_path: Path object for a file that is detected as a media file.
                media_type: One of "image", "video", "audio".
        """

        directory_path = Path(root_path)

        try:
            if isinstance(root_path, str) and root_path.strip() == "":
                raise ValueError("Provided path is empty or whitespace.")

            if not directory_path.exists():
                raise FileNotFoundError(f"Path does not exist: {directory_path}")

            if not directory_path.is_dir():
                raise NotADirectoryError(f"Path is not a directory: {directory_path}")

            if excluded_dirs is None:
                excluded_dirs = self.DEFAULT_EXCLUDES

            excludes = set(excluded_dirs) if excluded_dirs else self.DEFAULT_EXCLUDES

            # Walk through the given directory recursively
            for dirpath, dir_names, filenames in os.walk(directory_path):
                dir_names[:] = [dir_name for dir_name in dir_names if dir_name not in excludes and not dir_name.startswith('.')]

                for filename in filenames:
                    if filename.startswith("."):
                        continue

                    full_file_path = Path(dirpath) / filename
                    file_media_type = self._get_media_type(full_file_path.suffix)

                    if file_media_type:
                        stat = full_file_path.stat()
                        last_modified_dt = datetime.datetime.fromtimestamp(stat.st_mtime)

                        yield MediaAsset(
                            file_path=full_file_path,
                            media_type=file_media_type,
                            file_size_bytes=stat.st_size,
                            last_modified=last_modified_dt
                        )

        except (ValueError, NotADirectoryError, FileNotFoundError, PermissionError, IsADirectoryError,
                OSError) as e:
            print(f"[ERROR:{type(e).__name__}] Cannot read '{directory_path}': {e}")
            return

        except Exception as e:
            print(f"[ERROR:{type(e).__name__}] Cannot read '{directory_path}': {e}")
            return

if __name__ == "__main__":
    scanner = LibraryScanner()
    file_path = ''

    for media_path, media_type in scanner.scan(file_path):
        print(f"[{media_type.upper()}] found: {media_path}")