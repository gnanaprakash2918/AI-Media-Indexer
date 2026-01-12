"""Utilities for parsing subtitle files into normalized dialogue segments.

This module provides helper functions for reading and parsing SRT subtitle
files into structured segment dictionaries containing timing and text data.
The parsed output is designed to integrate directly with the VectorDB layer
used by the media ingestion pipeline.

Each subtitle segment includes:
- Dialogue text.
- Start and end timestamps in seconds.
- A fixed segment type (``"dialogue"``).

The module is intentionally lightweight and fault-tolerant: malformed or
unreadable subtitle files result in an empty segment list rather than
raising exceptions.
"""

import re
from pathlib import Path
from typing import Any

from core.utils.logger import log


def parse_srt(file_path: str | Path) -> list[dict[str, Any]]:
    """Parse an SRT subtitle file into normalized dialogue segments.

    The parsed output is suitable for insertion into the Vector DB and
    contains timing information in seconds along with the subtitle text.

    Args:
        file_path: Path to the `.srt` subtitle file.

    Returns:
        A list of subtitle segment dictionaries. Each dictionary contains:
        - ``text``: The subtitle text.
        - ``start``: Start time in seconds.
        - ``end``: End time in seconds.
        - ``type``: Segment type (always ``"dialogue"``).
    """
    path = Path(file_path)
    if not path.exists():
        return []

    try:
        content = path.read_text(encoding="utf-8", errors="ignore")
    except Exception as exc:  # noqa: BLE001
        log(f"[ERROR] Failed to read SRT file {path}: {exc}")
        return []

    pattern = re.compile(
        r"(\d+)\s*\n"
        r"(\d{2}:\d{2}:\d{2},\d{3})\s*-->\s*"
        r"(\d{2}:\d{2}:\d{2},\d{3})\s*\n"
        r"((?:(?!\n\d+\s*\n).)*)",
        re.DOTALL,
    )

    segments: list[dict[str, Any]] = []
    for match in pattern.finditer(content):
        start_str = match.group(2).replace(",", ".")
        end_str = match.group(3).replace(",", ".")
        text = match.group(4).strip()

        if not text:
            continue

        start = _timestamp_to_seconds(start_str)
        end = _timestamp_to_seconds(end_str)

        segments.append(
            {
                "text": text,
                "start": start,
                "end": end,
                "type": "dialogue",
            }
        )

    return segments


def _timestamp_to_seconds(timestamp: str) -> float:
    """Convert an SRT timestamp into seconds.

    Args:
        timestamp: Timestamp string in the form ``HH:MM:SS.mmm``.

    Returns:
        The timestamp converted to seconds as a float. Returns ``0.0`` if
        parsing fails.
    """
    try:
        hours, minutes, seconds = timestamp.split(":")
        return float(hours) * 3600 + float(minutes) * 60 + float(seconds)
    except (ValueError, IndexError):
        return 0.0
