"""Content moderation for detecting inappropriate content.

Provides detection of:
- NSFW visual content
- Violence/gore
- Hate speech (text)
- Sensitive content flags
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass
from enum import Enum
from typing import TYPE_CHECKING, Any, cast

from config import settings
from core.utils.logger import get_logger
from core.utils.resource_arbiter import RESOURCE_ARBITER

if TYPE_CHECKING:
    import numpy as np

log = get_logger(__name__)


class ContentFlag(Enum):
    """Content moderation flags."""

    SAFE = "safe"
    NSFW = "nsfw"
    VIOLENCE = "violence"
    GORE = "gore"
    HATE_SPEECH = "hate_speech"
    HARASSMENT = "harassment"
    SELF_HARM = "self_harm"
    SEXUAL = "sexual"
    DRUGS = "drugs"
    WEAPONS = "weapons"


@dataclass
class ModerationResult:
    """Result from content moderation."""

    is_safe: bool
    flags: list[ContentFlag]
    confidence: float
    details: dict[str, float]


class VisualContentModerator:
    """Detect inappropriate visual content.

    Uses NSFW detection models to flag sensitive frames.
    """

    def __init__(self, threshold: float = 0.7):
        """Initialize moderator.

        Args:
            threshold: Confidence threshold for flagging (0-1).
        """
        self.threshold = threshold
        self._model = None

    def cleanup(self):
        """Unload the model to free up resources."""
        if self._model is not None:
            log.info("[Moderation] Unloading NSFW model.")
            self._model = None

    async def _lazy_load(self) -> bool:
        """Load moderation model lazily in a thread."""
        if self._model is not None:
            return True

        try:
            from transformers import pipeline

            def _load():
                return pipeline(
                    "image-classification",
                    model="Falconsai/nsfw_image_detection",
                    device=-1,
                )

            # Move blocking download/load to thread
            self._model = await asyncio.to_thread(_load)

            # Register with Arbiter for OOM protection
            RESOURCE_ARBITER.register_model("moderation", self.cleanup)

            log.info("[Moderation] NSFW model loaded")
            return True
        except ImportError:
            log.warning("[Moderation] transformers not installed")
            return False
        except Exception as e:
            log.error(f"[Moderation] Model load failed: {e}")
            return False

    async def check_frame(self, frame: np.ndarray) -> ModerationResult:
        """Check a frame for inappropriate content.

        Args:
            frame: RGB frame as numpy array.

        Returns:
            ModerationResult with flags and confidence.
        """
        # Redundant check for safety
        if not getattr(settings, "enable_content_moderation", False):
            return ModerationResult(
                is_safe=True,
                flags=[ContentFlag.SAFE],
                confidence=0.0,
                details={},
            )

        if not await self._lazy_load():
            return ModerationResult(
                is_safe=True,
                flags=[ContentFlag.SAFE],
                confidence=0.0,
                details={},
            )

        if self._model is None:
            return ModerationResult(
                is_safe=True,
                flags=[ContentFlag.SAFE],
                confidence=0.0,
                details={"error": -1.0},
            )

        # Acquire VRAM budget (~1.0GB)
        async with RESOURCE_ARBITER.acquire("moderation", vram_gb=1.0):
            # Double check model exists (in case it was unloaded)
            if self._model is None:
                if not await self._lazy_load():
                    return ModerationResult(
                        is_safe=True,
                        flags=[ContentFlag.SAFE],
                        confidence=0.0,
                        details={"error": -1.0},
                    )

            try:
                from PIL import Image

                img = Image.fromarray(frame)

                # Run inference in thread to prevent blocking main loop
                raw_result = await asyncio.to_thread(self._model, img)

                # Pylance considers pipeline output as Generator/Iterable, forcing list cast for subscripting
                result = cast(list[dict[str, Any]], raw_result)

                flags = []
                details = {}
                nsfw_score = 0.0

                for item in result:
                    label = item["label"].lower()
                    score = item["score"]
                    details[label] = score

                    if "nsfw" in label or "porn" in label:
                        nsfw_score = max(nsfw_score, score)
                    if "safe" not in label and "normal" not in label:
                        if score > self.threshold:
                            flags.append(ContentFlag.NSFW)

                is_safe = nsfw_score < self.threshold
                if is_safe:
                    flags = [ContentFlag.SAFE]

                return ModerationResult(
                    is_safe=is_safe,
                    flags=flags,
                    confidence=nsfw_score,
                    details=details,
                )

            except Exception as e:
                log.error(f"[Moderation] Check failed: {e}")
                return ModerationResult(
                    is_safe=True,
                    flags=[ContentFlag.SAFE],
                    confidence=0.0,
                    details={"error": -1.0},
                )


class TextContentModerator:
    """Detect inappropriate text content.

    Uses toxicity classification for text moderation.
    """

    def __init__(self, threshold: float = 0.7):
        """Initialize text moderator.

        Args:
            threshold: Confidence threshold for flagging.
        """
        self.threshold = threshold
        self._pipeline = None

    async def _lazy_load(self) -> bool:
        """Load toxicity model lazily."""
        if self._pipeline is not None:
            return True

        try:
            from transformers import pipeline

            self._pipeline = pipeline(
                "text-classification",
                model="unitary/toxic-bert",
                device=-1,
            )
            log.info("[TextMod] Toxicity model loaded")
            return True
        except ImportError:
            log.warning("[TextMod] transformers not installed")
            return False
        except Exception as e:
            log.error(f"[TextMod] Model load failed: {e}")
            return False

    async def check_text(self, text: str) -> ModerationResult:
        """Check text for inappropriate content."""
        if not getattr(settings, "enable_content_moderation", False):
            return ModerationResult(
                is_safe=True,
                flags=[ContentFlag.SAFE],
                confidence=0.0,
                details={},
            )

        if not text or not await self._lazy_load():
            return ModerationResult(
                is_safe=True,
                flags=[ContentFlag.SAFE],
                confidence=0.0,
                details={},
            )

        if self._pipeline is None:
            return ModerationResult(
                is_safe=True,
                flags=[ContentFlag.SAFE],
                confidence=0.0,
                details={"error": -1.0},
            )

        try:
            raw_result = self._pipeline(text[:512])
            result = cast(list[dict[str, Any]], raw_result)[0]
            label = result["label"].lower()
            score = result["score"]

            flags = []
            is_safe = True

            if "toxic" in label and score > self.threshold:
                flags.append(ContentFlag.HATE_SPEECH)
                is_safe = False
            else:
                flags = [ContentFlag.SAFE]

            return ModerationResult(
                is_safe=is_safe,
                flags=flags,
                confidence=score if not is_safe else 1 - score,
                details={"label": label, "score": score},
            )

        except Exception as e:
            log.error(f"[TextMod] Check failed: {e}")
            return ModerationResult(
                is_safe=True,
                flags=[ContentFlag.SAFE],
                confidence=0.0,
                details={"error": -1.0},
            )
