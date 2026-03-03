"""Pipeline for visual grounding and segment tracking."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np

from core.tracking.sam3_tracker import SAM3Tracker
from core.utils.logger import get_logger

if TYPE_CHECKING:
    from core.storage.db import VectorDB

log = get_logger(__name__)


class GroundingPipeline:
    """Post-processing pipeline for visual grounding using SAM 3.

    Generates segmentation masks (masklets) for concepts in video frames.
    Designed to run offline/asynchronously.
    """

    def __init__(self, db: VectorDB | None = None) -> None:
        """Initialize the grounding pipeline.

        Args:
            db: Injected VectorDB instance. Created lazily if not provided.
        """
        self.sam = SAM3Tracker()
        self._db = db

    @property
    def db(self) -> VectorDB:
        """Lazy accessor for VectorDB."""
        if self._db is None:
            from core.storage.db import VectorDB
            self._db = VectorDB()
        return self._db

    async def process_video(
        self, video_path: str, concepts: list[str] | None = None
    ) -> int:
        """Run grounding on a video.

        Args:
            video_path: Path to the video file.
            concepts: Optional list of concepts to track. If None, extracts from metadata.

        Returns:
            Number of masklets created.
        """
        path = Path(video_path)
        if not path.exists():
            log.error(f"Video not found: {path}")
            return 0

        if not concepts:
            log.warning(f"No concepts provided for grounding: {video_path}")
            return 0

        log.info(
            f"Starting grounding for {path.name} with concepts: {concepts}"
        )

        try:
            import cv2
            cap = cv2.VideoCapture(str(path))
            fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
        except ImportError:
            fps = 25.0
            cap = None

        count = 0
        # SAM3Tracker manages its own GPU locking via ResourceArbiter.
        # No outer GPU_SEMAPHORE needed here.
        try:
            try:
            for concept in concepts:
                log.info(f"Tracking concept: {concept}")
                segments = await self.sam.track_concept(str(path), concept)

                for seg in segments:
                    frame_idx = seg.get("frame_idx", 0)
                    mask = seg.get("mask")

                    if mask is None:
                        continue

                    if cap is not None:
                        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
                        timestamp = cap.get(cv2.CAP_PROP_POS_MSEC) / 1000.0
                    else:
                        timestamp = frame_idx / fps

                    y_indices, x_indices = np.where(mask)
                    if len(y_indices) == 0:
                        continue

                    h, w = mask.shape
                    y_min, y_max = y_indices.min(), y_indices.max()
                    x_min, x_max = x_indices.min(), x_indices.max()

                    bbox_norm = [
                        int(x_min * 1000 / w),
                        int(y_min * 1000 / h),
                        int(x_max * 1000 / w),
                        int(y_max * 1000 / h),
                    ]

                    visual_vector = await self.sam.extract_visual_embedding(
                        str(path), mask, frame_idx
                    )

                    self.db.insert_masklet(
                        video_path=str(path),
                        concept=concept,
                        start_time=timestamp,
                        end_time=timestamp + (1.0 / fps),
                        confidence=seg.get("score", 1.0),
                        payload={
                            "bbox": bbox_norm,
                            "frame_idx": int(frame_idx),
                            "source": "sam3_grounding",
                        },
                        embedding=visual_vector,
                    )
                    count += 1

        except Exception as e:
            log.error(f"Grounding failed: {e}")

        if cap is not None:
            cap.release()

        log.info(f"Grounding complete. Created {count} masklets.")
        return count
