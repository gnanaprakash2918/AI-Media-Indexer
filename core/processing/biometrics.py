"""Biometric verification using ArcFace for identity arbitration."""

import numpy as np
import cv2
from pathlib import Path
from typing import TYPE_CHECKING

try:
    import onnxruntime as ort
except ImportError:
    ort = None

from core.utils.logger import log

if TYPE_CHECKING:
    from numpy.typing import NDArray


class BiometricArbitrator:
    """ArcFace-based biometric verification for twin disambiguation."""
    
    def __init__(self, model_path: Path | None = None):
        from config import settings
        self.model_path = model_path or settings.arcface_model_path
        self.threshold = settings.biometric_threshold
        self.session: "ort.InferenceSession | None" = None
        self._initialized = False
        
    def _lazy_load(self) -> bool:
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
            self.session = ort.InferenceSession(
                str(self.model_path), 
                providers=['CUDAExecutionProvider', 'CPUExecutionProvider']
            )
            log(f"[BIO] ArcFace loaded: {self.model_path.name}")
            return True
        except Exception as e:
            log(f"[BIO] Load failed: {e}")
            return False
    
    def get_embedding(self, face_crop: "NDArray[np.uint8]") -> "NDArray[np.float32] | None":
        if not self._lazy_load() or self.session is None:
            return None
            
        try:
            img = cv2.resize(face_crop, (112, 112))
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img = np.transpose(img, (2, 0, 1))
            img = np.expand_dims(img, 0)
            img = (img.astype(np.float32) - 127.5) / 128.0
            
            outputs = self.session.run(None, {self.session.get_inputs()[0].name: img})
            embedding = outputs[0][0]
            
            norm = np.linalg.norm(embedding)
            if norm > 0:
                embedding = embedding / norm
            return embedding.astype(np.float32)
        except Exception as e:
            log(f"[BIO] Embedding failed: {e}")
            return None
    
    def compute_similarity(self, emb1: "NDArray[np.float32]", emb2: "NDArray[np.float32]") -> float:
        norm1, norm2 = np.linalg.norm(emb1), np.linalg.norm(emb2)
        if norm1 < 1e-6 or norm2 < 1e-6:
            return 0.0
        return float(np.dot(emb1, emb2) / (norm1 * norm2))
    
    def verify_identity(self, emb1: "NDArray[np.float32] | None", emb2: "NDArray[np.float32] | None") -> bool:
        if emb1 is None or emb2 is None:
            return False
        similarity = self.compute_similarity(emb1, emb2)
        return similarity > (1.0 - self.threshold)
    
    def find_matching_identity(
        self,
        query_embedding: "NDArray[np.float32]",
        known_identities: dict[int, "NDArray[np.float32]"],
    ) -> int | None:
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
    global _arbitrator
    if _arbitrator is None:
        _arbitrator = BiometricArbitrator()
    return _arbitrator
