"""Emotion and Sentiment detection for video/audio content.

Provides analysis of:
- Facial emotions (happy, sad, angry, surprise, fear, disgust, neutral)
- Audio sentiment (positive, negative, neutral)
- Text sentiment from transcripts
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import TYPE_CHECKING

from core.utils.logger import get_logger

if TYPE_CHECKING:
    import numpy as np

log = get_logger(__name__)


class FacialEmotion(Enum):
    """Detected facial emotions."""

    HAPPY = "happy"
    SAD = "sad"
    ANGRY = "angry"
    SURPRISE = "surprise"
    FEAR = "fear"
    DISGUST = "disgust"
    NEUTRAL = "neutral"
    CONTEMPT = "contempt"


class Sentiment(Enum):
    """Overall sentiment classification."""

    POSITIVE = "positive"
    NEGATIVE = "negative"
    NEUTRAL = "neutral"


@dataclass
class EmotionResult:
    """Result from emotion detection."""

    emotion: FacialEmotion
    confidence: float
    face_box: tuple[int, int, int, int] | None = None
    timestamp: float = 0.0


@dataclass
class SentimentResult:
    """Result from sentiment analysis."""

    sentiment: Sentiment
    confidence: float
    scores: dict[str, float] = field(default_factory=dict)


class FacialEmotionDetector:
    """Detect emotions from facial expressions.

    Uses FER (Facial Expression Recognition) or DeepFace.
    """

    def __init__(self, use_gpu: bool = True):
        """Initialize emotion detector.

        Args:
            use_gpu: Whether to use GPU acceleration.
        """
        self.use_gpu = use_gpu
        self._detector = None
        self._fer = None

    async def _lazy_load(self) -> bool:
        """Load emotion model lazily."""
        if self._fer is not None:
            return True

        try:
            from fer import FER  # type: ignore

            self._fer = FER(mtcnn=True)
            log.info("[Emotion] FER model loaded")
            return True
        except ImportError:
            log.warning("[Emotion] fer library not installed")

        # Fallback to DeepFace
        try:
            from deepface import DeepFace

            self._detector = DeepFace
            log.info("[Emotion] Using DeepFace for emotions")
            return True
        except ImportError:
            log.warning("[Emotion] deepface not installed")
            return False

    async def detect(
        self,
        frame: np.ndarray,
        face_box: tuple[int, int, int, int] | None = None,
    ) -> list[EmotionResult]:
        """Detect emotions in a frame.

        Args:
            frame: RGB frame as numpy array.
            face_box: Optional specific face region (x, y, w, h).

        Returns:
            List of EmotionResult for each detected face.
        """
        if not await self._lazy_load():
            return []

        results = []

        try:
            if self._fer:
                emotions = self._fer.detect_emotions(frame)
                for face in emotions:
                    box = face.get("box", (0, 0, 0, 0))
                    emotion_scores = face.get("emotions", {})
                    top_emotion = max(emotion_scores, key=emotion_scores.get)
                    confidence = emotion_scores[top_emotion]

                    try:
                        emotion_enum = FacialEmotion(top_emotion.lower())
                    except ValueError:
                        emotion_enum = FacialEmotion.NEUTRAL

                    results.append(
                        EmotionResult(
                            emotion=emotion_enum,
                            confidence=confidence,
                            face_box=tuple(box),
                        )
                    )

            elif self._detector:
                analysis = self._detector.analyze(
                    frame,
                    actions=["emotion"],
                    enforce_detection=False,
                )
                if isinstance(analysis, list):
                    for face in analysis:
                        # Pylance sometimes sees 'face' as compatible with list if analysis type is vague
                        # Cast to dict to be safe
                        face_dict = face if isinstance(face, dict) else {}
                        emotion_data = face_dict.get("emotion", {})
                        top_emotion = face_dict.get(
                            "dominant_emotion", "neutral"
                        )
                        confidence = emotion_data.get(top_emotion, 0) / 100.0

                        try:
                            emotion_enum = FacialEmotion(top_emotion.lower())
                        except ValueError:
                            emotion_enum = FacialEmotion.NEUTRAL

                        results.append(
                            EmotionResult(
                                emotion=emotion_enum,
                                confidence=confidence,
                            )
                        )

            log.debug(f"[Emotion] Detected {len(results)} faces")
            return results

        except Exception as e:
            log.error(f"[Emotion] Detection failed: {e}")
            return []


class TextSentimentAnalyzer:
    """Analyze sentiment from text/transcripts.

    Uses transformers sentiment pipeline.
    """

    def __init__(
        self,
        model_name: str = "distilbert-base-uncased-finetuned-sst-2-english",
    ):
        """Initialize sentiment analyzer.

        Args:
            model_name: HuggingFace model for sentiment.
        """
        self.model_name = model_name
        self._pipeline = None

    async def _lazy_load(self) -> bool:
        """Load sentiment model lazily."""
        if self._pipeline is not None:
            return True

        try:
            from transformers import pipeline

            self._pipeline = pipeline(
                "sentiment-analysis",
                model=self.model_name,
                device=-1,  # CPU
            )
            log.info(f"[Sentiment] Loaded model: {self.model_name}")
            return True
        except ImportError:
            log.warning("[Sentiment] transformers not installed")
            return False
        except Exception as e:
            log.error(f"[Sentiment] Model load failed: {e}")
            return False

    async def analyze(self, text: str) -> SentimentResult:
        """Analyze sentiment of text.

        Args:
            text: Text to analyze.

        Returns:
            SentimentResult with classification and confidence.
        """
        if not text or not await self._lazy_load():
            return SentimentResult(
                sentiment=Sentiment.NEUTRAL,
                confidence=0.0,
            )

        if self._pipeline is None:
            return SentimentResult(sentiment=Sentiment.NEUTRAL, confidence=0.0)

        try:
            from typing import Any, cast

            # Cast pipeline output to list to avoid Generator inference issues
            raw_result = self._pipeline(text[:512])
            result_list = cast(list[dict[str, Any]], raw_result)
            result = result_list[0]  # Truncate to max length
            label = result["label"].lower()
            score = result["score"]

            if "positive" in label:
                sentiment = Sentiment.POSITIVE
            elif "negative" in label:
                sentiment = Sentiment.NEGATIVE
            else:
                sentiment = Sentiment.NEUTRAL

            return SentimentResult(
                sentiment=sentiment,
                confidence=score,
                scores={"raw_label": label, "raw_score": score},
            )

        except Exception as e:
            log.error(f"[Sentiment] Analysis failed: {e}")
            return SentimentResult(
                sentiment=Sentiment.NEUTRAL,
                confidence=0.0,
            )

    async def analyze_batch(self, texts: list[str]) -> list[SentimentResult]:
        """Analyze sentiment of multiple texts.

        Args:
            texts: List of texts to analyze.

        Returns:
            List of SentimentResult.
        """
        results = []
        for text in texts:
            result = await self.analyze(text)
            results.append(result)
        return results
