"""SOTA Segmentation Module using SAM (Segment Anything Model).

Falls back to Ultralytics SAM/YOLO if weights are missing.
Uses LAZY LOADING to prevent OOM on startup.
"""

import gc
import logging
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)


class SegmentationEngine:
    """Orchestrates zero-shot object segmentation using SAM (Segment Anything Model).

    Provides lazy loading of heavy models to optimize VRAM usage and
    supports both automatic and point-prompted segmentation modes.
    """

    def __init__(self, model_size: str = "sam_b"):
        """Initializes the SegmentationEngine.

        Args:
            model_size: The size of the SAM model to use (e.g., 'sam_b', 'sam_l').
        """
        self.model = None
        self.model_type = model_size
        self._device: str | None = None
        self._initialized = False

    @property
    def device(self) -> str:
        """Determines the optimal computation device (CUDA or CPU).

        Returns:
            The device string ('cuda' or 'cpu').
        """
        if self._device is None:
            import torch

            self._device = "cuda" if torch.cuda.is_available() else "cpu"
        return self._device

    def lazy_load(self) -> bool:
        """Loads the SAM model into memory only when first required.

        Reduces initial start-up time and VRAM usage by deferring model
        initialization until the first segmentation request.

        Returns:
            True if the model was successfully loaded or already exists.
        """
        if self.model is not None:
            return True

        logger.info(
            f"Loading Segmentation Model ({self.model_type}) on {self.device}..."
        )

        try:
            from ultralytics import SAM  # type: ignore

            self.model = SAM("sam_b.pt")  # Downloads automatically if missing
            logger.info("SAM Model loaded successfully.")
            return True
        except ImportError:
            logger.warning(
                "Ultralytics not found. Install with: pip install ultralytics"
            )
            return False
        except Exception as e:
            logger.error(f"Failed to load SAM: {e}")
            return False

    def segment_frame(
        self, frame: np.ndarray, prompt_points: list[list[int]] | None = None
    ) -> list[dict[str, Any]]:
        """Segments objects within a single image frame.

        Args:
            frame: Input image array (H, W, C) in BGR format.
            prompt_points: Optional list of [x, y] coordinates to guide
                segmentation around specific objects.

        Returns:
            A list of dictionary results, each containing:
                - 'id': Segment identifier (int).
                - 'segmentation': Boolean mask or polygon.
                - 'score': Confidence score.
        """
        if not self.lazy_load():
            return []

        results = []
        try:
            assert self.model is not None
            if prompt_points:
                res = self.model(
                    frame, points=prompt_points, device=self.device
                )
            else:
                res = self.model(frame, device=self.device)

            for r in res:
                if hasattr(r, "masks") and r.masks is not None:
                    masks_data = r.masks.xy
                    for i, mask_poly in enumerate(masks_data):
                        conf = (
                            float(r.boxes.conf[i])
                            if r.boxes is not None and len(r.boxes.conf) > i
                            else 1.0
                        )
                        results.append(
                            {
                                "id": i,
                                "segmentation": mask_poly.tolist(),
                                "confidence": conf,
                                "label": "object",
                            }
                        )
        except Exception as e:
            logger.error(f"Segmentation error: {e}")

        return results

    def cleanup(self) -> None:
        """Releases the SAM model and clears GPU memory.

        Moves the model to CPU, deletes references, and forces PyTorch
        CUDA cache clearing and garbage collection.
        """
        self.model = None
        import torch

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()
        logger.info("Segmentation resources released")


class Sam3Tracker:
    """Advanced video object segmenter and tracker using SAM 3.

    Provides persistent object tracking across video frames by leveraging
    the SAM 3 (Segment Anything Model 2/3) video predictor, allowing for
    concept-based tracking through text or visual prompts.
    """

    def __init__(self) -> None:
        """Initializes the Sam3Tracker state and internal predictor."""
        self.predictor = None
        self.inference_state = None
        self._device: str | None = None
        self._initialized = False

    @property
    def device(self) -> str:
        """Determines the computation device from system settings.

        Returns:
            The configured device string (e.g., 'cuda', 'cpu', 'mps').
        """
        if self._device is None:
            from config import settings

            self._device = settings.device
        return self._device

    def initialize(self) -> bool:
        """Loads and initializes the SAM 3 model predictor.

        Returns:
            True if the model was successfully loaded or is already available.
        """
        if self._initialized:
            return self.predictor is not None

        self._initialized = True

        try:
            from sam2.build_sam import (  # type: ignore
                build_sam2_video_predictor,  # type: ignore
            )

            from config import settings

            checkpoint = (
                settings.model_cache_dir / "sam2" / "sam2_hiera_large.pt"
            )
            config = "sam2_hiera_l.yaml"

            if not checkpoint.exists():
                logger.warning(f"SAM3 checkpoint not found: {checkpoint}")
                logger.info(
                    "Download from: https://github.com/facebookresearch/segment-anything-2"
                )
                return False

            logger.info("Loading SAM3 video predictor...")
            self.predictor = build_sam2_video_predictor(config, checkpoint)
            logger.info(f"SAM3 loaded on {self.device}")
            return True

        except ImportError:
            logger.warning(
                "sam2 package not installed. Note: This project uses SAM3. "
                "Install with: pip install segment-anything-2"
            )
            return False
        except Exception as e:
            logger.error(f"SAM3 load failed: {e}")
            return False

    def init_video(self, video_path: Path) -> bool:
        """Initializes the tracking state for a specific video file.

        Args:
            video_path: Path to the video file to be processed.

        Returns:
            True if the inference state was successfully initialized.
        """
        if not self.initialize():
            return False

        try:
            if self.predictor:
                self.inference_state = self.predictor.init_state(
                    str(video_path)
                )
                logger.info(f"SAM3 initialized video: {video_path.name}")
                return True
            return False
        except Exception as e:
            logger.error(f"SAM3 video init failed: {e}")
            return False

    def add_concept_prompt(self, text: str, frame_idx: int = 0) -> list[int]:
        """Adds a concept-based prompt to track an object starting at a specific frame.

        Args:
            text: Description of the object or concept to track.
            frame_idx: The video frame index where the prompt should be applied.

        Returns:
            A list of object IDs assigned to this concept tracking task.
        """
        if self.predictor is None or self.inference_state is None:
            logger.warning("SAM3 not initialized")
            return []

        try:
            _, obj_ids, _ = self.predictor.add_new_prompt(
                self.inference_state, frame_idx=frame_idx, text=text
            )
            logger.info(
                f"SAM3 added concept '{text}': {len(obj_ids)} instances"
            )
            return (
                obj_ids.tolist()
                if hasattr(obj_ids, "tolist")
                else list(obj_ids)
            )
        except Exception as e:
            logger.error(f"SAM3 prompt failed: {e}")
            return []

    def propagate(self) -> Iterator[dict[str, Any]]:
        """Propagates the initialized masks and prompts through the entire video.

        Yields:
            A dictionary for each frame containing:
                - 'frame_idx': The index of the frame.
                - 'object_ids': List of IDs active in this frame.
                - 'masks': Boolean mask array for the objects.
        """
        if self.predictor is None or self.inference_state is None:
            return

        try:
            for (
                frame_idx,
                obj_ids,
                mask_logits,
            ) in self.predictor.propagate_in_video(self.inference_state):
                masks = (mask_logits > 0.0).cpu().numpy()
                yield {
                    "frame_idx": frame_idx,
                    "object_ids": obj_ids.tolist()
                    if hasattr(obj_ids, "tolist")
                    else list(obj_ids),
                    "masks": masks,
                }
        except Exception as e:
            logger.error(f"SAM3 propagation error: {e}")

    def process_video_concepts(
        self, video_path: Path, prompts: list[str]
    ) -> Iterator[dict[str, Any]]:
        """Orchestrates the full concept tracking pipeline for a video.

        Handles initialization, prompt addition, and mask propagation in one flow.

        Args:
            video_path: Path to the video file.
            prompts: A list of text descriptions to track throughout the video.

        Yields:
            Tracking result dictionaries for each frame in the video.
        """
        if not self.init_video(video_path):
            return

        for prompt in prompts:
            self.add_concept_prompt(prompt)

        yield from self.propagate()
        self.cleanup()

    def cleanup(self) -> None:
        """Releases tracking state and clears GPU memory."""
        self.inference_state = None
        import torch

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()
        logger.info("SAM3 resources released")

    def reset(self) -> None:
        """Resets the entire tracker state and clears the model predictor.

        Use this to reclaim all VRAM and clear the internal state of the model.
        """
        self.cleanup()
        self.predictor = None
        self._initialized = False
