"""Stub module for voice clustering.

.. deprecated::
    Voice clustering is not yet implemented. This module exists only as a
    placeholder so that the API route ``/voices/cluster`` returns a clear
    "not implemented" response instead of a 500 error.  When voice clustering
    is implemented, replace this stub with the real logic.
"""

from __future__ import annotations

from typing import Any

from core.utils.logger import get_logger

log = get_logger(__name__)

if False:  # TYPE_CHECKING
    from core.storage.db import VectorDB


async def cluster_voices(db: "VectorDB") -> dict[str, Any]:
    """Stub for voice clustering — returns a clear 'not implemented' response.

    Args:
        db: The VectorDB instance (unused in stub).

    Returns:
        A status dictionary indicating the feature is not yet implemented.
    """
    log.warning("[VoiceClustering] cluster_voices called but not implemented")
    return {
        "status": "not_implemented",
        "num_clusters": 0,
        "message": "Voice clustering not yet implemented. "
                   "Use identity-based speaker diarization via /voices endpoints instead.",
    }
