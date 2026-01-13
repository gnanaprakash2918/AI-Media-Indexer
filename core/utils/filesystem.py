"""Windows-safe path handling with Unicode and long path support.

Fixes "Windows Unicode path normalization broke ingest" bug.
Handles paths >260 chars, Indic scripts, emoji in filenames.
"""

import os
import re
import unicodedata
from pathlib import Path


def safe_path(path_str: str | Path) -> Path:
    """Convert path string to a Windows-safe Path object.

    Handles:
    - Long paths (>260 chars) via \\\\?\\ prefix
    - Unicode normalization (NFC)
    - Path separator normalization

    Args:
        path_str: Input path string or Path object.

    Returns:
        Safe Path object that works on Windows/NTFS.

    Example:
        >>> safe_path("📽️Brahmāstra_தமிழ்.mp4")
        WindowsPath('\\\\?\\C:\\...\\📽️Brahmāstra_தமிழ்.mp4')
    """
    if isinstance(path_str, Path):
        path_str = str(path_str)

    # Normalize Unicode to NFC (composed form)
    path_str = unicodedata.normalize("NFC", path_str)

    if os.name == "nt":  # Windows
        # Convert forward slashes to backslashes
        path_str = path_str.replace("/", "\\")

        # Get absolute path
        abs_path = os.path.abspath(path_str)

        # Prepend \\?\ for long path support (if not already present)
        if not abs_path.startswith("\\\\?\\"):
            if abs_path.startswith("\\\\"):  # UNC path
                abs_path = "\\\\?\\UNC\\" + abs_path[2:]
            else:
                abs_path = "\\\\?\\" + abs_path

        return Path(abs_path)

    # Non-Windows: just normalize
    return Path(path_str)


def sanitize_filename(filename: str, max_length: int = 200) -> str:
    """Sanitize a filename for cross-platform compatibility.

    Args:
        filename: Original filename.
        max_length: Maximum length (default 200 to leave room for path).

    Returns:
        Safe filename that works on Windows, macOS, Linux.

    Example:
        >>> sanitize_filename('file:with*bad?chars.mp4')
        'file_with_bad_chars.mp4'
    """
    # Normalize Unicode
    filename = unicodedata.normalize("NFC", filename)

    # Remove/replace Windows-illegal characters: \ / : * ? " < > |
    illegal_chars = r'[\\/:*?"<>|]'
    filename = re.sub(illegal_chars, "_", filename)

    # Remove control characters
    filename = "".join(c for c in filename if unicodedata.category(c) != "Cc")

    # Trim to max length (preserving extension)
    if len(filename) > max_length:
        name, ext = os.path.splitext(filename)
        max_name_len = max_length - len(ext)
        filename = name[:max_name_len] + ext

    # Handle reserved Windows names
    reserved = {
        "CON", "PRN", "AUX", "NUL",
        "COM1", "COM2", "COM3", "COM4", "COM5", "COM6", "COM7", "COM8", "COM9",
        "LPT1", "LPT2", "LPT3", "LPT4", "LPT5", "LPT6", "LPT7", "LPT8", "LPT9",
    }
    name_upper = os.path.splitext(filename)[0].upper()
    if name_upper in reserved:
        filename = "_" + filename

    return filename.strip(". ")  # Remove leading/trailing dots and spaces


def ensure_parent_exists(path: Path | str) -> Path:
    """Ensure parent directory exists, with Windows-safe handling.

    Args:
        path: Path whose parent should exist.

    Returns:
        Safe Path object with parent directory created.
    """
    safe_p = safe_path(path)
    safe_p.parent.mkdir(parents=True, exist_ok=True)
    return safe_p


def is_safe_filename(filename: str) -> bool:
    """Check if a filename is safe for cross-platform use.

    Args:
        filename: Filename to check.

    Returns:
        True if filename is safe.
    """
    sanitized = sanitize_filename(filename)
    return sanitized == filename and len(filename) > 0
