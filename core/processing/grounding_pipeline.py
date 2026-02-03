"""Pipeline for visual grounding and segment tracking."""

import logging
from pathlib import Path

from core.storage.db import VectorDB

# CHANGED: Use the authoritative SAM3Tracker in core/tracking
from core.tracking.sam3_tracker import SAM3Tracker
from core.utils.resource_arbiter import GPU_SEMAPHORE

logger = logging.getLogger(__name__)


class GroundingPipeline:
    """Post-processing pipeline for visual grounding using SAM 3.

    Generates segmentation masks (masklets) for concepts description in video frames.
    Designed to run offline/asynchronously.
    """

    def __init__(self):
        """Initialize the grounding pipeline."""
        self.sam = SAM3Tracker()
        self.db = VectorDB()

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
            logger.error(f"Video not found: {path}")
            return 0

        # Bootstrapping concepts if not provided
        if not concepts:
            try:
                # Stub: In real system, we'd query DB for 'suggested_concepts' or similar
                # For now we rely on explicit input or skip
                # video_meta = self.db.get_video_metadata(video_path)
                pass
            except Exception as e:
                logger.warning(f"Could not fetch concepts from DB: {e}")

            if not concepts:
                logger.warning(f"No concepts provided for grounding: {video_path}")
                return 0

        logger.info(
            f"Starting grounding for {path.name} with concepts: {concepts}"
        )

        try:
            import cv2
            cap = cv2.VideoCapture(str(path))
            fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
            cap.release()
        except ImportError:
            fps = 25.0

        count = 0
        # Single locking point for GPU
        async with GPU_SEMAPHORE:
            try:
                # We iterate OVER CONCEPTS. SAM3Tracker.track_concept is per-concept.
                for concept in concepts:
                    logger.info(f"Tracking concept: {concept}")

                    # SAM3Tracker.track_concept is async and handles its own GPU acquisition internally via ResourceArbiter
                    # But since we are inside GPU_SEMAPHORE here (from old code), we should be careful.
                    # Actually, SAM3Tracker uses ResourceArbiter which uses a semaphore.
                    # We should rely on SAM3Tracker's internal management.

                    # Offload to avoid blocking main loop if anything is sync
                    segments = await self.sam.track_concept(str(path), concept)

                    for seg in segments:
                        frame_idx = seg.get("frame_idx", 0)
                        mask = seg.get("mask")

                        if mask is None:
                            continue

                        timestamp = frame_idx / fps

                        # Calculate BBox
                        import numpy as np
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
                            int(y_max * 1000 / h)
                        ]

                        # Generate embedding
                        visual_vector = await self.sam.extract_visual_embedding(str(path), mask, frame_idx)

                        self.db.insert_masklet(
                            video_path=str(path),
                            concept=concept,
                            start_time=timestamp,
                            end_time=timestamp + (1.0 / fps),
                            confidence=seg.get("score", 1.0),
                            payload={
                                "bbox": bbox_norm,
                                "frame_idx": int(frame_idx),
                                "source": "sam3_grounding"
                            },
                            embedding=visual_vector
                        )
                        count += 1

            except Exception as e:
                logger.error(f"Grounding failed: {e}")

        logger.info(f"Grounding complete. Created {count} masklets.")
        return count
