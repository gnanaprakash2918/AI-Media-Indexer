"""Biometric verification using ArcFace for identity arbitration."""

from pathlib import Path
from typing import TYPE_CHECKING, Any

import cv2
import numpy as np

try:
    import onnxruntime as ort
except ImportError:
    ort = None

import asyncio
from core.utils.logger import log
from core.utils.resource_arbiter import GPU_SEMAPHORE

if TYPE_CHECKING:
    from numpy.typing import NDArray


class BiometricArbitrator:
    """ArcFace-based biometric verification for twin disambiguation."""

    def __init__(self, model_path: Path | None = None) -> None:
        """Initializes the BiometricArbitrator.

        Args:
            model_path: Optional path to the ArcFace ONNX model. If None,
                it is loaded from system settings.
        """
        from config import settings

        self.model_path = model_path or settings.arcface_model_path
        self.threshold = settings.biometric_threshold
        self.session: Any | None = None
        self._initialized = False

    async def _lazy_load(self) -> bool:
        """Loads the ArcFace ONNX session only when first required."""
        if self._initialized:
            return self.session is not None

        async with asyncio.Lock():  # Basic lock for init
            if self._initialized:
                return self.session is not None
            self._initialized = True

            if ort is None:
                log("[BIO] onnxruntime not installed")
                return False

            if not self.model_path.exists():
                log(f"[BIO] ArcFace model not found: {self.model_path}")
                return False

            try:

                def _load():
                    return ort.InferenceSession(
                        str(self.model_path),
                        providers=[
                            "CUDAExecutionProvider",
                            "CPUExecutionProvider",
                        ],
                    )

                self.session = await asyncio.to_thread(_load)
                log(f"[BIO] ArcFace loaded: {self.model_path.name}")
                return True
            except Exception as e:
                log(f"[BIO] Load failed: {e}")
                return False

    async def get_embedding(
        self, face_crop: "NDArray[np.uint8]"
    ) -> "NDArray[np.float32] | None":
        """Extracts a 128/512D ArcFace embedding from a face crop."""
        if not await self._lazy_load() or self.session is None:
            return None

        try:

            def _prep():
                img = cv2.resize(face_crop, (112, 112))
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                img = np.transpose(img, (2, 0, 1))
                img = np.expand_dims(img, 0)
                return (img.astype(np.float32) - 127.5) / 128.0

            img_input = await asyncio.to_thread(_prep)

            async with GPU_SEMAPHORE:

                def _infer():
                    outputs = self.session.run(
                        None, {self.session.get_inputs()[0].name: img_input}
                    )
                    embedding = outputs[0][0]
                    norm = np.linalg.norm(embedding)
                    if norm > 0:
                        embedding = embedding / norm
                    return embedding.astype(np.float32)

                return await asyncio.to_thread(_infer)
        except Exception as e:
            log(f"[BIO] Embedding failed: {e}")
            return None

    def compute_similarity(
        self, emb1: "NDArray[np.float32]", emb2: "NDArray[np.float32]"
    ) -> float:
        """Calculates the cosine similarity between two embeddings.

        Args:
            emb1: First feature vector.
            emb2: Second feature vector.

        Returns:
            Similarity score between 0.0 and 1.0.
        """
        norm1, norm2 = np.linalg.norm(emb1), np.linalg.norm(emb2)
        if norm1 < 1e-6 or norm2 < 1e-6:
            return 0.0
        return float(np.dot(emb1, emb2) / (norm1 * norm2))

    def verify_identity(
        self,
        emb1: "NDArray[np.float32] | None",
        emb2: "NDArray[np.float32] | None",
    ) -> bool:
        """Verifies if two embeddings belong to the same identity.

        Args:
            emb1: Query embedding.
            emb2: Reference embedding.

        Returns:
            True if the similarity exceeds the configured threshold.
        """
        if emb1 is None or emb2 is None:
            return False
        similarity = self.compute_similarity(emb1, emb2)
        return similarity > (1.0 - self.threshold)

    def find_matching_identity(
        self,
        query_embedding: "NDArray[np.float32]",
        known_identities: dict[int, "NDArray[np.float32]"],
    ) -> int | None:
        """Finds the best matching identity from a set of known embeddings.

        Args:
            query_embedding: The embedding to match.
            known_identities: A dictionary mapping cluster IDs to embeddings.

        Returns:
            The cluster ID of the best match, or None if no match meets the threshold.
        """
        best_match_id = None
        best_similarity = 0.0
        threshold = 1.0 - self.threshold

        for cluster_id, known_emb in known_identities.items():
            similarity = self.compute_similarity(query_embedding, known_emb)
            if similarity > threshold and similarity > best_similarity:
                best_similarity = similarity
                best_match_id = cluster_id

        return best_match_id


_arbitrator: BiometricArbitrator | None = None


def get_biometric_arbitrator() -> BiometricArbitrator:
    """Retrieves the singleton BiometricArbitrator instance.

    Returns:
        The initialized BiometricArbitrator.
    """
    global _arbitrator
    if _arbitrator is None:
        _arbitrator = BiometricArbitrator()
    return _arbitrator
