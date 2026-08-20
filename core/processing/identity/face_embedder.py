"""Face embedding utilities."""

import hashlib
import numpy as np
from pathlib import Path
from typing import Any, Sequence
from numpy.typing import NDArray, ArrayLike
from config import settings

CACHE_DIR = settings.project_root() / ".face_cache"

class FaceEmbedder:
    def __init__(self):
        CACHE_DIR.mkdir(parents=True, exist_ok=True)

    def _cache_key(
        self, image: NDArray[Any], box: tuple[int, int, int, int], version: str
    ) -> str:
        return f"{hashlib.sha1(image.data[:2048]).hexdigest()}_{box}_{version}"

    def _disk_cache_get(self, key: str) -> NDArray[np.float64] | None:
        p = CACHE_DIR / f"{key}.npy"
        return np.load(p) if p.exists() else None

    def _disk_cache_put(self, key: str, val: NDArray[np.float64]) -> None:
        p = CACHE_DIR / f"{key}.npy"
        if not p.exists():
            np.save(p, val)

    def _to_2d_array(self, encodings: Sequence[ArrayLike]) -> NDArray[np.float64]:
        arrs = [np.asarray(e, dtype=np.float64) for e in encodings]
        if len({a.shape for a in arrs}) > 1:
            raise ValueError("Inconsistent shapes")
        return np.vstack(arrs)
