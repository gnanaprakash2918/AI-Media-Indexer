"""Object detection using YOLO-World for attribute-based queries.

Enables searching for objects with specific attributes like
"person wearing blue t-shirt" or "red car near building".
"""

from __future__ import annotations

import asyncio
from typing import Any

import numpy as np

from core.utils.logger import get_logger

log = get_logger(__name__)


class ObjectDetector:
    """YOLO-World open-vocabulary object detection.

    Detects objects with natural language descriptions, enabling
    attribute-based queries like "blue t-shirt" or "bowling pins".

    Usage:
        detector = ObjectDetector()
        objects = await detector.detect(frame, classes=["blue shirt", "car"])
        # [{"class": "blue shirt", "confidence": 0.85, "bbox": [...]}]
    """

    def __init__(
        self,
        model_size: str = "m",
        device: str | None = None,
    ):
        """Initialize object detector.

        Args:
            model_size: Model size ('s', 'm', 'l', 'x'). Default 'm'.
            device: Device to run on. Auto-detected if None.
        """
        self.model_size = model_size
        self._device = device
        self.model = None
        self._init_lock = asyncio.Lock()

    def _get_device(self) -> str:
        """Get device to use."""
        if self._device:
            return self._device
        try:
            import torch
            return "cuda" if torch.cuda.is_available() else "cpu"
        except ImportError:
            return "cpu"

    async def _lazy_load(self) -> bool:
        """Load YOLO-World model lazily.

        Returns:
            True if model loaded successfully.
        """
        if self.model is not None:
            return True

        async with self._init_lock:
            if self.model is not None:
                return True

            try:
                from core.utils.resource_arbiter import RESOURCE_ARBITER

                async with RESOURCE_ARBITER.acquire("yolo_world", vram_gb=1.0):
                    log.info(
                        f"[YOLO-World] Loading model size={self.model_size}"
                    )

                    from ultralytics import YOLO

                    model_name = f"yolov8{self.model_size}-worldv2.pt"
                    self.model = YOLO(model_name)

                    device = self._get_device()
                    self._device = device

                    log.info(f"[YOLO-World] Model loaded on {device}")
                    return True

            except ImportError as e:
                log.warning(f"[YOLO-World] ultralytics not available: {e}")
                return False
            except Exception as e:
                log.error(f"[YOLO-World] Failed to load: {e}")
                return False

    async def detect(
        self,
        frame: np.ndarray,
        classes: list[str],
        confidence: float = 0.3,
    ) -> list[dict[str, Any]]:
        """Detect objects matching given classes.

        Args:
            frame: RGB/BGR frame as numpy array.
            classes: List of class descriptions to detect.
            confidence: Minimum detection confidence.

        Returns:
            List of detections with class, confidence, and bbox.
        """
        if not await self._lazy_load():
            return []

        if not classes:
            return []

        try:
            from core.utils.resource_arbiter import RESOURCE_ARBITER

            async with RESOURCE_ARBITER.acquire("yolo_world", vram_gb=1.0):
                # Set custom classes for zero-shot detection
                self.model.set_classes(classes)

                # Run inference
                results = self.model.predict(
                    frame,
                    conf=confidence,
                    verbose=False,
                    device=self._device,
                )

                detections = []
                for result in results:
                    boxes = result.boxes
                    if boxes is None:
                        continue

                    for i in range(len(boxes)):
                        cls_id = int(boxes.cls[i])
                        conf = float(boxes.conf[i])
                        bbox = boxes.xyxy[i].tolist()

                        detections.append({
                            "class": classes[cls_id] if cls_id < len(classes) else "unknown",
                            "confidence": round(conf, 3),
                            "bbox": bbox,  # [x1, y1, x2, y2]
                            "class_id": cls_id,
                        })

                if detections:
                    log.debug(f"[YOLO-World] Detected {len(detections)} objects")

                return detections

        except Exception as e:
            log.error(f"[YOLO-World] Detection failed: {e}")
            return []

    async def detect_common_objects(
        self,
        frame: np.ndarray,
        confidence: float = 0.3,
    ) -> list[dict[str, Any]]:
        """Detect common objects using default COCO classes.

        Args:
            frame: Input frame.
            confidence: Minimum confidence.

        Returns:
            List of detected objects.
        """
        common_classes = [
            "person", "car", "dog", "cat", "chair", "table",
            "phone", "laptop", "book", "cup", "bottle",
            "bicycle", "motorcycle", "bus", "truck",
            "bird", "horse", "sheep", "cow", "elephant",
            "tv", "keyboard", "mouse", "remote", "clock",
        ]
        return await self.detect(frame, common_classes, confidence)

    def cleanup(self) -> None:
        """Release model resources."""
        if self.model is not None:
            del self.model
            self.model = None

        try:
            import torch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except ImportError:
            pass

        log.info("[YOLO-World] Resources released")


class GroundingDINODetector:
    """Alternative detector using Grounding DINO (more accurate).

    Better for complex natural language queries like
    "person in red shirt standing near a blue car".
    """

    def __init__(self, device: str | None = None):
        """Initialize Grounding DINO detector.

        Args:
            device: Device to run on. Auto-detected if None.
        """
        self._device = device
        self.model = None
        self.processor = None
        self._init_lock = asyncio.Lock()

    async def _lazy_load(self) -> bool:
        """Load Grounding DINO model."""
        if self.model is not None:
            return True

        async with self._init_lock:
            if self.model is not None:
                return True

            try:
                from core.utils.resource_arbiter import RESOURCE_ARBITER

                async with RESOURCE_ARBITER.acquire("grounding_dino", vram_gb=2.0):
                    log.info("[GroundingDINO] Loading model...")

                    from transformers import (
                        AutoModelForZeroShotObjectDetection,
                        AutoProcessor,
                    )

                    model_id = "IDEA-Research/grounding-dino-tiny"
                    self.processor = AutoProcessor.from_pretrained(model_id)
                    self.model = AutoModelForZeroShotObjectDetection.from_pretrained(
                        model_id
                    )

                    import torch
                    device = self._device or (
                        "cuda" if torch.cuda.is_available() else "cpu"
                    )
                    self.model.to(device)
                    self._device = device

                    log.info(f"[GroundingDINO] Model loaded on {device}")
                    return True

            except ImportError as e:
                log.warning(f"[GroundingDINO] transformers not available: {e}")
                return False
            except Exception as e:
                log.error(f"[GroundingDINO] Failed to load: {e}")
                return False

    async def detect(
        self,
        frame: np.ndarray,
        query: str,
        threshold: float = 0.3,
    ) -> list[dict[str, Any]]:
        """Detect objects matching natural language query.

        Args:
            frame: Input frame.
            query: Natural language query (e.g., "red car. blue shirt.").
            threshold: Detection threshold.

        Returns:
            List of detections.
        """
        if not await self._lazy_load():
            return []

        try:
            import torch
            from PIL import Image
            from core.utils.resource_arbiter import RESOURCE_ARBITER

            async with RESOURCE_ARBITER.acquire("grounding_dino", vram_gb=2.0):
                # Convert to PIL Image
                if isinstance(frame, np.ndarray):
                    image = Image.fromarray(frame)
                else:
                    image = frame

                # Process inputs
                inputs = self.processor(
                    images=image,
                    text=query,
                    return_tensors="pt",
                )
                inputs = {k: v.to(self._device) for k, v in inputs.items()}

                with torch.no_grad():
                    outputs = self.model(**inputs)

                # Post-process
                results = self.processor.post_process_grounded_object_detection(
                    outputs,
                    inputs["input_ids"],
                    box_threshold=threshold,
                    text_threshold=threshold,
                    target_sizes=[image.size[::-1]],
                )

                detections = []
                if results:
                    result = results[0]
                    boxes = result["boxes"].tolist()
                    scores = result["scores"].tolist()
                    labels = result["labels"]

                    for box, score, label in zip(boxes, scores, labels):
                        detections.append({
                            "class": label,
                            "confidence": round(score, 3),
                            "bbox": box,
                        })

                return detections

        except Exception as e:
            log.error(f"[GroundingDINO] Detection failed: {e}")
            return []

    def cleanup(self) -> None:
        """Release resources."""
        if self.model:
            del self.model
            self.model = None
        if self.processor:
            del self.processor
            self.processor = None

        try:
            import torch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except ImportError:
            pass

        log.info("[GroundingDINO] Resources released")
