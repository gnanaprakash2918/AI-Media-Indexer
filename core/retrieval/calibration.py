from __future__ import annotations


def normalize_vector_score(cosine_sim: float) -> float:
    MIN_EXPECTED = 0.3
    MAX_EXPECTED = 0.85
    clamped = max(MIN_EXPECTED, min(MAX_EXPECTED, cosine_sim))
    normalized = (clamped - MIN_EXPECTED) / (MAX_EXPECTED - MIN_EXPECTED)
    return round(normalized, 4)


def normalize_vlm_score(vlm_confidence: int) -> float:
    return round(vlm_confidence / 100.0, 4)


def combine_scores(vector_score: float, vlm_confidence: int | None) -> float:
    if vlm_confidence is None:
        return normalize_vector_score(vector_score)
    vec_norm = normalize_vector_score(vector_score)
    vlm_norm = normalize_vlm_score(vlm_confidence)
    combined = (vec_norm * 0.2) + (vlm_norm * 0.8)
    return round(combined, 4)


def generate_thumbnail_url(video_path: str, timestamp: float) -> str:
    from urllib.parse import quote

    encoded_path = quote(video_path, safe="")
    return f"/api/media/thumbnail?path={encoded_path}&time={timestamp:.2f}"
