from __future__ import annotations

import asyncio
from pathlib import Path
from typing import Any

import numpy as np

from core.utils.logger import get_logger

log = get_logger(__name__)


class ActiveSpeakerDetector:
    def __init__(self):
        self._face_detector = None
        self._lip_model = None
        self._device = None
        self._init_lock = asyncio.Lock()

    def _get_device(self) -> str:
        if self._device:
            return self._device
        try:
            import torch
            return "cuda" if torch.cuda.is_available() else "cpu"
        except ImportError:
            return "cpu"

    async def _lazy_load(self) -> bool:
        if self._lip_model is not None:
            return True

        async with self._init_lock:
            if self._lip_model is not None:
                return True
            try:
                log.info("[ActiveSpeaker] Using motion-based lip detection (lightweight)")
                self._lip_model = "motion_based"
                self._device = self._get_device()
                log.info(f"[ActiveSpeaker] Initialized on {self._device}")
                return True
            except Exception as e:
                log.error(f"[ActiveSpeaker] Init failed: {e}")
                return False

    def _extract_lip_region(
        self,
        frame: np.ndarray,
        face_bbox: tuple[int, int, int, int],
    ) -> np.ndarray | None:
        x1, y1, x2, y2 = face_bbox
        face_height = y2 - y1
        face_width = x2 - x1

        lip_y1 = y1 + int(face_height * 0.6)
        lip_y2 = y2
        lip_x1 = x1 + int(face_width * 0.2)
        lip_x2 = x2 - int(face_width * 0.2)

        h, w = frame.shape[:2]
        lip_y1 = max(0, lip_y1)
        lip_y2 = min(h, lip_y2)
        lip_x1 = max(0, lip_x1)
        lip_x2 = min(w, lip_x2)

        if lip_y2 <= lip_y1 or lip_x2 <= lip_x1:
            return None

        return frame[lip_y1:lip_y2, lip_x1:lip_x2]

    async def detect_lip_motion(
        self,
        frame1: np.ndarray,
        frame2: np.ndarray,
        face_bbox: tuple[int, int, int, int],
    ) -> dict[str, Any]:
        if not await self._lazy_load():
            return {"is_speaking": False, "error": "Model not loaded"}

        try:
            import cv2

            lip1 = self._extract_lip_region(frame1, face_bbox)
            lip2 = self._extract_lip_region(frame2, face_bbox)

            if lip1 is None or lip2 is None:
                return {"is_speaking": False, "error": "Could not extract lip region"}

            if lip1.shape != lip2.shape:
                lip2 = cv2.resize(lip2, (lip1.shape[1], lip1.shape[0]))

            if len(lip1.shape) == 3:
                lip1_gray = cv2.cvtColor(lip1, cv2.COLOR_RGB2GRAY)
                lip2_gray = cv2.cvtColor(lip2, cv2.COLOR_RGB2GRAY)
            else:
                lip1_gray = lip1
                lip2_gray = lip2

            diff = cv2.absdiff(lip1_gray, lip2_gray)
            motion_score = float(np.mean(diff))

            motion_threshold = 5.0
            is_speaking = motion_score > motion_threshold

            log.debug(f"[ActiveSpeaker] Lip motion score: {motion_score:.2f}, speaking: {is_speaking}")

            return {
                "is_speaking": is_speaking,
                "motion_score": motion_score,
                "threshold": motion_threshold,
                "confidence": min(motion_score / 20.0, 1.0) if is_speaking else 0.0,
            }

        except Exception as e:
            log.error(f"[ActiveSpeaker] Detection failed: {e}")
            return {"is_speaking": False, "error": str(e)}

    async def detect_speaking_faces(
        self,
        frame1: np.ndarray,
        frame2: np.ndarray,
        face_bboxes: list[tuple[int, int, int, int]],
    ) -> list[dict[str, Any]]:
        results = []
        for i, bbox in enumerate(face_bboxes):
            result = await self.detect_lip_motion(frame1, frame2, bbox)
            result["face_index"] = i
            result["bbox"] = bbox
            results.append(result)

        speaking_faces = [r for r in results if r.get("is_speaking")]
        log.info(f"[ActiveSpeaker] {len(speaking_faces)}/{len(face_bboxes)} faces speaking")

        return results

    async def matches_speaking_constraint(
        self,
        frame1: np.ndarray,
        frame2: np.ndarray,
        face_bbox: tuple[int, int, int, int],
        should_be_speaking: bool = True,
    ) -> dict[str, Any]:
        result = await self.detect_lip_motion(frame1, frame2, face_bbox)
        is_speaking = result.get("is_speaking", False)

        if should_be_speaking:
            result["matches"] = is_speaking
        else:
            result["matches"] = not is_speaking

        return result

    def cleanup(self) -> None:
        self._lip_model = None
        self._face_detector = None
        log.info("[ActiveSpeaker] Resources released")
