"""Stub module for voice clustering."""

from typing import Any

from core.storage.db import VectorDB


async def cluster_voices(db: VectorDB) -> dict[str, Any]:
    """Stub for voice clustering functionality.

    Args:
        db: The VectorDB instance.

    Returns:
        Statistics dictionary.
    """
    return {
        "status": "not_implemented",
        "num_clusters": 0,
        "message": "Voice clustering not yet implemented",
    }
