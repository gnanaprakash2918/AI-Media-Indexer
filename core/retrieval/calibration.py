"""Scoring and normalization utilities for cross-modal retrieval."""

from __future__ import annotations


def normalize_vector_score(cosine_sim: float) -> float:
    """Normalizes a raw cosine similarity score to a [0.0, 1.0] range.

    Args:
        cosine_sim: The raw vector similarity score.

    Returns:
        The normalized score rounded to 4 decimal places.
    """
    min_expected = 0.3
    max_expected = 0.85
    clamped = max(min_expected, min(max_expected, cosine_sim))
    normalized = (clamped - min_expected) / (max_expected - min_expected)
    return round(normalized, 4)


def normalize_vlm_score(vlm_confidence: int) -> float:
    """Converts an integer VLM confidence score [0-100] to a [0.0, 1.0] float.

    Args:
        vlm_confidence: Confidence score from the VLM analyzer.

    Returns:
        The normalized float score.
    """
    return round(vlm_confidence / 100.0, 4)


def combine_scores(vector_score: float, vlm_confidence: int | None) -> float:
    """Weighted fusión of vector similarity and VLM confidence.

    Prioritizes VLM confidence (80%) when available, falling back to
    pure vector distance otherwise.

    Args:
        vector_score: Raw cosine similarity from the vector database.
        vlm_confidence: Optional confidence score from structured VLM analysis.

    Returns:
        A combined score used for final search ranking.
    """
    if vlm_confidence is None:
        return normalize_vector_score(vector_score)
    vec_norm = normalize_vector_score(vector_score)
    vlm_norm = normalize_vlm_score(vlm_confidence)
    combined = (vec_norm * 0.2) + (vlm_norm * 0.8)
    return round(combined, 4)


def generate_thumbnail_url(video_path: str, timestamp: float) -> str:
    """Generates a backend API URL for a specific video frame thumbnail.

    Args:
        video_path: Absolute or relative path to the source video.
        timestamp: The exact time in seconds for the frame extraction.

    Returns:
        A formatted string URL for the UI to consume.
    """
    from urllib.parse import quote

    encoded_path = quote(video_path, safe="")
    return f"/api/media/thumbnail?path={encoded_path}&time={timestamp:.2f}"
