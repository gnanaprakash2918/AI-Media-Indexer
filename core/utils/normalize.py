"""Result normalization utilities for consistent API responses."""

from typing import Any
from urllib.parse import quote


def normalize_timestamp(result: dict[str, Any]) -> float:
    """Extract timestamp from result with fallback chain."""
    return float(
        result.get("start_time")
        or result.get("timestamp")
        or result.get("start")
        or 0
    )


def normalize_end_time(
    result: dict[str, Any], default_duration: float = 5.0
) -> float:
    """Extract end time from result."""
    start = normalize_timestamp(result)
    return float(
        result.get("end_time")
        or result.get("end")
        or (start + default_duration)
    )


def normalize_media_path(result: dict[str, Any]) -> str:
    """Extract media path from result."""
    return str(result.get("media_path") or result.get("video_path") or "")


def add_media_urls(
    result: dict[str, Any], base_url: str = ""
) -> dict[str, Any]:
    """Add thumbnail and playback URLs to result."""
    media = normalize_media_path(result)
    ts = normalize_timestamp(result)
    end_ts = normalize_end_time(result)

    if not media:
        return result

    safe_path = quote(str(media))

    result["thumbnail_url"] = (
        f"{base_url}/media/thumbnail?path={safe_path}&time={ts}"
    )
    result["playback_url"] = (
        f"{base_url}/media?path={safe_path}#t={max(0, ts - 3)}"
    )
    result["display_start"] = max(0, ts - 3)
    result["display_end"] = end_ts + 3
    result["match_start"] = ts
    result["match_end"] = end_ts

    return result


def normalize_result(
    result: dict[str, Any], base_url: str = ""
) -> dict[str, Any]:
    """Fully normalize a search result for frontend consumption."""
    normalized = {**result}
    normalized["media_path"] = normalize_media_path(result)
    normalized["video_path"] = normalized["media_path"]  # Backwards compat
    normalized["start_time"] = normalize_timestamp(result)
    normalized["timestamp"] = normalized["start_time"]  # Backwards compat
    normalized["end_time"] = normalize_end_time(result)

    if normalized["media_path"] and "thumbnail_url" not in normalized:
        normalized = add_media_urls(normalized, base_url)

    return normalized
