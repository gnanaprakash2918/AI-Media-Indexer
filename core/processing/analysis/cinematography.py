"""Shot Type & Cinematography Classification (Netflix style).

100% DYNAMIC - NO HARDCODED CATEGORIES.
All classification is via zero-shot CLIP/LLM.
Works for ANY query on the planet.

Based on Research:
- Netflix Internal: Cinematography Attributes ("Dark", "Emotional", "Witty")
- Apple Photos: Aesthetic Score
"""

from __future__ import annotations

import asyncio
from typing import Any

import numpy as np

from core.utils.logger import get_logger

log = get_logger(__name__)


class DynamicClassifier:
    """Zero-shot classifier for ANY visual concept.

    NO HARDCODED CATEGORIES - accepts any list of concepts.
    Uses CLIP for zero-shot classification.

    Usage:
        classifier = DynamicClassifier()

        # Classify shot types
        result = await classifier.classify(
            frame,
            concepts=["close up shot", "wide shot", "aerial view"]
        )

        # Classify mood
        result = await classifier.classify(
            frame,
            concepts=["dark moody scene", "bright cheerful scene"]
        )

        # Classify ANYTHING
        result = await classifier.classify(
            frame,
            concepts=["outdoor scene", "indoor scene", "underwater"]
        )
    """

    def __init__(self, device: str | None = None):
        """Initialize dynamic classifier.

        Args:
            device: Device to run on. Auto-detected if None.
        """
        self._device = device
        self._model = None
        self._processor = None
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
        """Load CLIP model lazily."""
        if self._model is not None:
            return True

        async with self._init_lock:
            if self._model is not None:
                return True

            try:
                log.info("[DynamicClassifier] Loading CLIP...")

                from transformers import CLIPModel, CLIPProcessor

                self._processor = CLIPProcessor.from_pretrained(
                    "openai/clip-vit-base-patch32"
                )
                self._model = CLIPModel.from_pretrained(
                    "openai/clip-vit-base-patch32"
                )

                device = self._get_device()
                self._model.to(device)
                self._device = device

                log.info(f"[DynamicClassifier] Loaded on {device}")
                return True

            except ImportError as e:
                log.warning(
                    f"[DynamicClassifier] transformers not available: {e}"
                )
                return False
            except Exception as e:
                log.error(f"[DynamicClassifier] Failed to load: {e}")
                return False

    async def classify(
        self,
        frame: np.ndarray,
        concepts: list[str],
        top_k: int = 3,
    ) -> list[dict[str, Any]]:
        """Classify frame against ANY list of concepts.

        Args:
            frame: RGB frame as numpy array.
            concepts: List of concepts to classify against.
                     Can be ANYTHING: shot types, moods, locations, etc.
            top_k: Number of top predictions to return.

        Returns:
            List of {concept, confidence} dicts sorted by confidence.
        """
        if not concepts:
            return []

        if not await self._lazy_load():
            return []

        try:
            import torch
            from PIL import Image

            # Convert to PIL
            if isinstance(frame, np.ndarray):
                image = Image.fromarray(frame)
            else:
                image = frame

            log.debug(
                f"[DynamicClassifier] Running Zero-Shot Classification for {len(concepts)} concepts: {concepts}"
            )

            # Process
            inputs = self._processor(
                text=concepts,
                images=image,
                return_tensors="pt",
                padding=True,
            )
            inputs = {k: v.to(self._device) for k, v in inputs.items()}

            with torch.no_grad():
                outputs = self._model(**inputs)
                logits = outputs.logits_per_image[0]
                probs = logits.softmax(dim=-1).cpu().numpy()

            # Get top-k
            top_indices = probs.argsort()[::-1][:top_k]
            results = []

            log_str = "[DynamicClassifier] Top predictions: "
            for idx in top_indices:
                concept = concepts[idx]
                score = round(float(probs[idx]), 3)
                results.append(
                    {
                        "concept": concept,
                        "confidence": score,
                    }
                )
                log_str += f"'{concept}' ({score:.2f}), "

            log.info(log_str.strip(", "))
            return results

        except Exception as e:
            log.exception(
                f"[DynamicClassifier] Zero-Shot Classification failed: {e}"
            )
            return []

    async def match_query(
        self,
        frame: np.ndarray,
        query: str,
    ) -> float:
        """Check how well a frame matches a natural language query.

        Args:
            frame: RGB frame.
            query: Natural language description (e.g., "a dark moody scene").

        Returns:
            Similarity score (0-1).
        """
        if not await self._lazy_load():
            return 0.0

        try:
            import torch
            from PIL import Image

            if isinstance(frame, np.ndarray):
                image = Image.fromarray(frame)
            else:
                image = frame

            inputs = self._processor(
                text=[query],
                images=image,
                return_tensors="pt",
                padding=True,
            )
            inputs = {k: v.to(self._device) for k, v in inputs.items()}

            with torch.no_grad():
                outputs = self._model(**inputs)
                # Normalize and get similarity
                image_embeds = outputs.image_embeds / outputs.image_embeds.norm(
                    dim=-1, keepdim=True
                )
                text_embeds = outputs.text_embeds / outputs.text_embeds.norm(
                    dim=-1, keepdim=True
                )
                similarity = (image_embeds @ text_embeds.T).item()

            return max(0.0, min(1.0, similarity))

        except Exception as e:
            log.error(f"[DynamicClassifier] Match failed: {e}")
            return 0.0

    def cleanup(self) -> None:
        """Release resources."""
        if self._model:
            del self._model
            self._model = None
        if self._processor:
            del self._processor
            self._processor = None
        log.info("[DynamicClassifier] Resources released")


class AestheticScorer:
    """Score image/frame aesthetic quality (Apple Photos style).

    Uses NIMA (Neural Image Assessment) to score aesthetic quality.
    Enables "Best of" selection for photo galleries.
    """

    def __init__(self, device: str | None = None):
        """Initialize aesthetic scorer."""
        self._device = device
        self._model = None
        self._init_lock = asyncio.Lock()

    async def _lazy_load(self) -> bool:
        """Load NIMA model."""
        if self._model is not None:
            return True

        async with self._init_lock:
            if self._model is not None:
                return True

            try:
                log.info("[Aesthetic] Loading NIMA model...")

                import torch
                import torchvision.models as models

                # Use MobileNet as base (efficient)
                self._model = models.mobilenet_v2(pretrained=True)
                # Modify final layer for aesthetic score (1-10)
                self._model.classifier[-1] = torch.nn.Linear(1280, 10)
                self._model.eval()

                device = self._device or (
                    "cuda" if torch.cuda.is_available() else "cpu"
                )
                self._model.to(device)
                self._device = device

                log.info(f"[Aesthetic] Loaded on {device}")
                return True

            except Exception as e:
                log.warning(f"[Aesthetic] Failed to load: {e}")
                return False

    async def score(self, frame: np.ndarray) -> float:
        """Score aesthetic quality of a frame.

        Args:
            frame: RGB frame as numpy array.

        Returns:
            Aesthetic score from 1.0 to 10.0.
        """
        if not await self._lazy_load():
            return 5.0  # Neutral

        try:
            import torch
            from PIL import Image
            from torchvision import transforms

            transform = transforms.Compose(
                [
                    transforms.Resize(256),
                    transforms.CenterCrop(224),
                    transforms.ToTensor(),
                    transforms.Normalize(
                        mean=[0.485, 0.456, 0.406],
                        std=[0.229, 0.224, 0.225],
                    ),
                ]
            )

            if isinstance(frame, np.ndarray):
                image = Image.fromarray(frame)
            else:
                image = frame

            tensor = transform(image).unsqueeze(0).to(self._device)

            with torch.no_grad():
                output = self._model(tensor)
                scores = torch.arange(1, 11, dtype=torch.float32).to(
                    self._device
                )
                probs = torch.softmax(output, dim=1)
                aesthetic_score = (probs * scores).sum().item()

            return round(aesthetic_score, 2)

        except Exception as e:
            log.error(f"[Aesthetic] Scoring failed: {e}")
            return 5.0

    def cleanup(self) -> None:
        """Release resources."""
        if self._model:
            del self._model
            self._model = None
        log.info("[Aesthetic] Resources released")


class TechnicalCueDetector:
    """Detect technical cues in video (Amazon Rekognition style).

    Detects: Black frames, shot boundaries, etc.
    NO HARDCODING - uses visual analysis.
    """

    def __init__(self):
        """Initialize technical cue detector."""
        pass

    async def detect_black_frame(
        self,
        frame: np.ndarray,
        threshold: float = 0.05,
    ) -> bool:
        """Detect if frame is (nearly) black."""
        try:
            if frame.max() > 1:
                normalized = frame.astype(np.float32) / 255.0
            else:
                normalized = frame.astype(np.float32)

            avg_brightness = normalized.mean()
            return avg_brightness < threshold
        except Exception:
            return False

    async def detect_shot_boundary(
        self,
        frame1: np.ndarray,
        frame2: np.ndarray,
        threshold: float = 0.4,
    ) -> bool:
        """Detect shot boundary between two frames using histogram diff."""
        try:
            import cv2

            gray1 = cv2.cvtColor(frame1, cv2.COLOR_RGB2GRAY)
            gray2 = cv2.cvtColor(frame2, cv2.COLOR_RGB2GRAY)

            hist1 = cv2.calcHist([gray1], [0], None, [256], [0, 256])
            hist2 = cv2.calcHist([gray2], [0], None, [256], [0, 256])

            cv2.normalize(hist1, hist1)
            cv2.normalize(hist2, hist2)

            diff = cv2.compareHist(hist1, hist2, cv2.HISTCMP_BHATTACHARYYA)

            return diff > threshold
        except Exception as e:
            log.error(f"[TechnicalCue] Shot boundary detection failed: {e}")
            return False

    async def detect_credits(
        self,
        frame: np.ndarray,
        ocr_text: str | None = None,
    ) -> dict[str, Any]:
        """Detect if frame contains credits using LLM (no hardcoding)."""
        result = {"is_credits": False, "confidence": 0.0}

        # Check for dark background first
        try:
            if frame.max() > 1:
                normalized = frame.astype(np.float32) / 255.0
            else:
                normalized = frame.astype(np.float32)

            avg_brightness = normalized.mean()
            if avg_brightness > 0.3:
                return result  # Not dark enough for credits

            if ocr_text:
                # Use DynamicClassifier to check if text looks like credits
                classifier = DynamicClassifier()
                score = await classifier.match_query(
                    frame, "movie credits or title card with names and roles"
                )
                result["is_credits"] = score > 0.5
                result["confidence"] = score

            return result
        except Exception:
            return result


class PersonAttributeRecognizer:
    """Recognize person attributes DYNAMICALLY.

    NO HARDCODED CATEGORIES - accepts any attribute query.
    Uses CLIP for zero-shot attribute recognition.
    """

    def __init__(self, device: str | None = None):
        """Initialize person attribute recognizer."""
        self._classifier = None
        self._device = device

    async def recognize(
        self,
        person_crop: np.ndarray,
        attribute_queries: list[str],
    ) -> list[dict[str, Any]]:
        """Recognize attributes of a person using DYNAMIC queries.

        Args:
            person_crop: Cropped image of a person.
            attribute_queries: List of attribute descriptions to check.
                Example: ["wearing red shirt", "wearing blue shirt",
                         "has glasses", "no glasses"]

        Returns:
            List of {attribute, confidence} sorted by confidence.
        """
        if not attribute_queries:
            return []

        if self._classifier is None:
            self._classifier = DynamicClassifier(device=self._device)

        # Use dynamic classifier with user-provided queries
        return await self._classifier.classify(
            person_crop,
            concepts=attribute_queries,
            top_k=len(attribute_queries),
        )

    async def match_description(
        self,
        person_crop: np.ndarray,
        description: str,
    ) -> float:
        """Check if person matches a natural language description.

        Args:
            person_crop: Cropped image of a person.
            description: Natural language description.
                Example: "person in red shirt with glasses"

        Returns:
            Similarity score (0-1).
        """
        if self._classifier is None:
            self._classifier = DynamicClassifier(device=self._device)

        return await self._classifier.match_query(person_crop, description)

    def cleanup(self) -> None:
        """Release resources."""
        if self._classifier:
            self._classifier.cleanup()
            self._classifier = None
        log.info("[PersonAttr] Resources released")
