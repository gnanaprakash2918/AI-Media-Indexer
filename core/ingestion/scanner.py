"""Scan directories for media files and yield media asset metadata."""

import datetime
import os
from collections.abc import Generator, Iterable
from pathlib import Path

from core.schemas import MediaAsset, MediaType


class LibraryScanner:
    """Scan directories for media files (images, audio, video)."""

    DEFAULT_EXCLUDES: set[str] = {
        ".git",
        ".hg",
        ".svn",
        ".venv",
        "venv",
        ".env",
        "env",
        "__pycache__",
        ".mypy_cache",
        ".pytest_cache",
        "node_modules",
        ".cache",
        ".local",
        ".Trash",
        "lost+found",
        ".DS_Store",
        "__MACOSX",
        "Thumbs.db",
        "desktop.ini",
        "dist",
        "build",
        ".egg-info",
        ".idea",
        ".vscode",
    }

    VIDEO_EXTS: set[str] = {".mp4", ".mkv", ".mov", ".avi", ".webm", ".m4v"}
    AUDIO_EXTS: set[str] = {".mp3", ".wav", ".aac", ".flac", ".m4a", ".ogg"}
    IMAGE_EXTS: set[str] = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tiff"}

    def _get_media_type(self, extension: str) -> MediaType | None:
        """Determine media type based on a file extension.

        Args:
          extension: File extension including the leading dot.

        Returns:
          A MediaType value if the extension is known; otherwise None.
        """
        extension = extension.lower()
        if extension in self.VIDEO_EXTS:
            return MediaType.VIDEO
        if extension in self.AUDIO_EXTS:
            return MediaType.AUDIO
        if extension in self.IMAGE_EXTS:
            return MediaType.IMAGE

        return None

    def scan(
        self,
        root_path: str | Path,
        excluded_dirs: Iterable[str] | None = None,
    ) -> Generator[MediaAsset, None, None]:
        """Recursively yield media files (images, audio, video).

        Args:
          root_path: Root directory to scan. Can be a string or a Path
            object.
          excluded_dirs: Iterable of directory or file names to exclude
            by name. If None, a default set of common project and system
            directories is used.

        Yields:
          MediaAsset instances for files detected as media files.
        """
        directory_path = Path(root_path)

        try:
            if isinstance(root_path, str) and root_path.strip() == "":
                raise ValueError("Provided path is empty or whitespace.")

            if not directory_path.exists():
                raise FileNotFoundError(
                    f"Path does not exist: {directory_path}"
                )

            if not directory_path.is_dir():
                raise NotADirectoryError(
                    f"Path is not a directory: {directory_path}"
                )

            if excluded_dirs is None:
                excluded_dirs = self.DEFAULT_EXCLUDES

            excludes = (
                set(excluded_dirs) if excluded_dirs else self.DEFAULT_EXCLUDES
            )

            # Walk through the given directory recursively.
            for dirpath, dir_names, filenames in os.walk(directory_path):
                dir_names[:] = [
                    dir_name
                    for dir_name in dir_names
                    if dir_name not in excludes and not dir_name.startswith(".")
                ]

                for filename in filenames:
                    if filename.startswith("."):
                        continue

                    full_file_path = Path(dirpath) / filename
                    file_media_type = self._get_media_type(
                        full_file_path.suffix
                    )

                    if file_media_type:
                        stat = full_file_path.stat()
                        last_modified_dt = datetime.datetime.fromtimestamp(
                            stat.st_mtime
                        )

                        yield MediaAsset(
                            file_path=full_file_path,
                            media_type=file_media_type,
                            file_size_bytes=stat.st_size,
                            last_modified=last_modified_dt,
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
                f"Cannot read '{directory_path}': {exc}"
            )
            return

        except Exception as exc:  # pylint: disable=broad-except
            print(
                f"[ERROR:{type(exc).__name__}] "
                f"Cannot read '{directory_path}': {exc}"
            )
            return


if __name__ == "__main__":
    scanner = LibraryScanner()
    file_path = ""

    for media_asset in scanner.scan(file_path):
        print(
            f"[{media_asset.media_type.value.upper()}] "
            f"found: {media_asset.file_path}"
        )
