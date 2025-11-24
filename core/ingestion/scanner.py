from pathlib import Path
from typing import Generator, Union, Iterable
import magic

class LibraryScanner:
    """System to efficiently find media files (images, audio, video) under a directory."""

    DEFAULT_EXCLUDES = {
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

        try:
            mime = magic.from_file(str(file_path), mime=True)
            if mime is None:
                return False, "none"

        except (FileNotFoundError, PermissionError, IsADirectoryError, OSError) as e:
            print(f"[ERROR:{type(e).__name__}] Cannot read '{file_path}': {e}")
            return False, "none"

        except Exception as e:
            print(f"[ERROR:{type(e).__name__}] Cannot read '{file_path}': {e}")
            return False, "none"

        for prefix, media_type_itr in (
                ("image/", "image"),
                ("video/", "video"),
                ("audio/", "audio"),
        ):
            if mime.startswith(prefix):
                return True, media_type_itr

        return False, "none"

    def scan(self, root_path: Union[str, Path], excluded_dirs_files: Iterable[str] | None = None) -> Generator[tuple[Path, str], None, None]:
        """
        Recursively yield media files (images, audio, video) under the given directory.

        Args:
            root_path: Root directory to scan. Can be a string or a Path object.
            excluded_dirs_files: Iterable of directory or file names to exclude by name. If None, a default set of common project/system directories is used.

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

        except (ValueError, NotADirectoryError, FileNotFoundError, PermissionError, IsADirectoryError, OSError) as e:
            print(f"[ERROR:{type(e).__name__}] Cannot read '{directory_path}': {e}")
            return

        except Exception as e:
            print(f"[ERROR:{type(e).__name__}] Cannot read '{directory_path}': {e}")
            return

        if excluded_dirs_files is None:
            excluded_dirs_files = self.DEFAULT_EXCLUDES

        excluded_set = set(excluded_dirs_files)

        # Walk through the given directory recursively
        for path in directory_path.rglob('*'):
            if not path.is_file():
                continue

            if any(part in excluded_set for part in path.parts):
                # Skip if it's mentioned in Excluded files
                continue

            is_media, file_type = self._is_media_file(path)

            if is_media:
                yield path, file_type


if __name__ == "__main__":
    scanner = LibraryScanner()
    file_path = ''

    for media_path, media_type in scanner.scan(file_path):
        print(f"[{media_type.upper()}] found: {media_path}")