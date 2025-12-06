import re
from pathlib import Path
from typing import Any


def parse_srt(file_path: str | Path) -> list[dict[str, Any]]:
    """Parses an SRT file into a list of segments for the Vector DB.

    Args:
        file_path: Path to the .srt file.

    Returns:
        A list of dictionaries with 'text', 'start', 'end', and 'type'.
    """
    path = Path(file_path)
    if not path.exists():
        return []

    try:
        with open(path, "r", encoding="utf-8", errors="ignore") as f:
            content = f.read()
    except Exception as e:
        print(f"[ERROR] Failed to read SRT file {path}: {e}")
        return []

    # Regex to find blocks: Number -> Time -> Text
    # Example: 00:00:20,000 --> 00:00:24,400
    # Group 1: Index
    # Group 2: Start Timestamp
    # Group 3: End Timestamp
    # Group 4: Subtitle Text (multiline until the next index block)
    pattern = re.compile(
        r"(\d+)\s*\n(\d{2}:\d{2}:\d{2},\d{3})\s*-->\s*(\d{2}:\d{2}:\d{2},\d{3})\s*\n((?:(?!\d+\s*\n).)*)",
        re.DOTALL,
    )

    segments = []
    for match in pattern.finditer(content):
        start_str = match.group(2).replace(",", ".")
        end_str = match.group(3).replace(",", ".")
        text = match.group(4).strip()

        start = _timestamp_to_seconds(start_str)
        end = _timestamp_to_seconds(end_str)

        if text:
            segments.append(
                {
                    "text": text,
                    "start": start,
                    "end": end,
                    "type": "dialogue",
                }
            )

    return segments


def _timestamp_to_seconds(ts: str) -> float:
    """Converts timestamp string (00:00:20.000) to float seconds (20.0)."""
    try:
        parts = ts.split(":")
        hours = float(parts[0])
        minutes = float(parts[1])
        seconds = float(parts[2])
        return hours * 3600 + minutes * 60 + seconds
    except (ValueError, IndexError):
        return 0.0
