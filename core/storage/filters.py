"""Qdrant filter builders for consistent query construction."""

from qdrant_client import models


def media_path_filter(path: str | None) -> models.Condition | None:
    """Build filter for media_path field."""
    if not path:
        return None
    return models.FieldCondition(
        key="media_path",
        match=models.MatchValue(value=path),
    )


def video_path_filter(path: str | None) -> models.Condition | None:
    """Build filter for video_path field (legacy collections)."""
    if not path:
        return None
    return models.FieldCondition(
        key="video_path",
        match=models.MatchValue(value=path),
    )


def face_cluster_filter(cluster_ids: list[int] | None) -> models.Condition | None:
    """Build filter for face_cluster_id field."""
    if not cluster_ids:
        return None
    return models.FieldCondition(
        key="face_cluster_id",
        match=models.MatchAny(any=cluster_ids),
    )


def voice_cluster_filter(cluster_ids: list[int] | None) -> models.Condition | None:
    """Build filter for voice_cluster_id field."""
    if not cluster_ids:
        return None
    return models.FieldCondition(
        key="voice_cluster_id",
        match=models.MatchAny(any=cluster_ids),
    )


def scan_id_filter(scan_id: str | None) -> models.Condition | None:
    """Build filter for scan_id field."""
    if not scan_id:
        return None
    return models.FieldCondition(
        key="scan_id",
        match=models.MatchValue(value=scan_id),
    )


def text_contains_filter(field: str, text: str) -> models.Condition:
    """Build text contains filter."""
    return models.FieldCondition(
        key=field,
        match=models.MatchText(text=text),
    )


def build_filter(conditions: list[models.Condition | None]) -> models.Filter | None:
    """Combine multiple conditions into a Filter with must clause."""
    valid = [c for c in conditions if c is not None]
    if not valid:
        return None
    return models.Filter(must=valid)


def build_should_filter(conditions: list[models.Condition | None]) -> models.Filter | None:
    """Combine conditions with OR logic (should clause)."""
    valid = [c for c in conditions if c is not None]
    if not valid:
        return None
    return models.Filter(should=valid)
