from pathlib import Path
from typing import Generator
import magic

class LibraryScanner:
    """system to find all files efficiently."""

    def _is_media_file(self, file_path: Path) -> bool:
        """ Return True if the file is an image, audio, or video based on MIME type."""
        try:
            mime = magic.from_file(str(file_path), mime=True)
            if mime is None:
                return False

        except FileNotFoundError:
            print(f"File not found: {file_path}")
            return False

        except PermissionError:
            print(f"Insufficient Permissions :  Denied {file_path}")
            return False

        except IsADirectoryError:
            print(f"Path is a directory, not a file : {file_path}")
            return False

        except OSError:
            print(f"IO Error when reading this file path : {file_path}")
            return False

        except Exception as e:
            print(f"Error: {e}")
            return False

        return (
                mime.startswith('image/') or
                mime.startswith('video/') or
                mime.startswith('audio/')
        )

    def scan(self, root_path: str) -> Generator[Path, None, None]:
        """
        Recursively yield media files (images, audio, video) under the given directory.

        Args:
            directory: Root directory to scan.

        Yields:
            Path objects for files that are detected as media files.
        """

        directory_path = Path(root_path)

        try:
            if not directory_path or root_path.strip() == '':
                raise ValueError("Provided path is empty or whitespace.")

            if not directory_path.exists():
                raise FileNotFoundError(f"Path does not exist: {directory_path}")

            if not directory_path.is_dir():
                raise NotADirectoryError(f"Path is not a directory: {directory_path}")

        except ValueError as e:
            print(f"[ERROR] Invalid path: {e}")
            return

        except FileNotFoundError as e:
            print(f"[ERROR] {e}")
            return

        except NotADirectoryError as e:
            print(f"[ERROR] {e}")
            return

        except OSError as e:
            print(f"[ERROR] OS error while accessing '{directory_path}': {e}")
            return

        # Walk through the given directory recursively
        for path in directory_path.rglob('*'):
            if not path.is_file():
                continue

            if self._is_media_file(path):
                yield path