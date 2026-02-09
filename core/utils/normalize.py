"""Result normalization utilities for consistent API responses."""

from typing import Any
from urllib.parse import quote

from config import settings


def normalize_timestamp(result: dict[str, Any]) -> float:
    """Extract timestamp from result with fallback chain.
    
    Priority: start_time > timestamp > start > 0
    Note: Uses explicit None checks to handle valid 0.0 timestamps correctly.
    """
    # Try each key in priority order (explicit None check, not truthy)
    for key in ("start_time", "timestamp", "start"):
        val = result.get(key)
        if val is not None:
            try:
                return float(val)
            except (TypeError, ValueError):
                continue  # Skip invalid values
    return 0.0


def normalize_end_time(result: dict[str, Any]) -> float:
    """Extract end time from result, using configurable default duration.
    
    Priority: end_time > end > (start + default_duration)
    """
    start = normalize_timestamp(result)
    
    # Try each key in priority order (explicit None check)
    for key in ("end_time", "end"):
        val = result.get(key)
        if val is not None:
            try:
                return float(val)
            except (TypeError, ValueError):
                continue
    
    return start + settings.search_default_duration


def normalize_media_path(result: dict[str, Any]) -> str:
    """Extract media path from result."""
    return str(result.get("media_path") or result.get("video_path") or "")


def add_media_urls(result: dict[str, Any], base_url: str = "") -> dict[str, Any]:
    """Add thumbnail and playback URLs to result with configurable padding."""
    media = normalize_media_path(result)
    ts = normalize_timestamp(result)
    end_ts = normalize_end_time(result)

    if not media:
        return result

    safe_path = quote(str(media))

    # Use configurable padding from settings
    display_start = max(0, ts - settings.search_padding_before)
    display_end = end_ts + settings.search_padding_after

    result["thumbnail_url"] = f"{base_url}/media/thumbnail?path={safe_path}&time={ts}"
    result["playback_url"] = f"{base_url}/media?path={safe_path}#t={display_start}"
    result["display_start"] = display_start
    result["display_end"] = display_end
    result["match_start"] = ts
    result["match_end"] = end_ts

    return result



def normalize_result(result: dict[str, Any], base_url: str = "") -> dict[str, Any]:
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
