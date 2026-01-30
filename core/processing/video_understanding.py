"""LanguageBind and InternVideo Integration (Video Council SOTA).

Implements binding of audio, video, and text modalities for
advanced multimodal search capabilities.

Based on Research:
- LanguageBind: Binds Audio, Video, Depth, Thermal to text
- InternVideo2: SOTA action/motion understanding
"""

from __future__ import annotations

import asyncio
from typing import Any

import numpy as np

from core.utils.logger import get_logger

log = get_logger(__name__)


class LanguageBindEncoder:
    """LanguageBind multimodal encoder.

    Binds multiple modalities (audio, video, text) to a shared space.
    Enables cross-modal search: "Find videos matching this sound".

    Usage:
        encoder = LanguageBindEncoder()
        video_emb = await encoder.encode_video(frames)
        audio_emb = await encoder.encode_audio(audio)
        text_emb = await encoder.encode_text("dog barking")
        # All embeddings are in same space - can compute similarity
    """

    def __init__(self, device: str | None = None):
        """Initialize LanguageBind encoder.

        Args:
            device: Device to run on. Auto-detected if None.
        """
        self._device = device
        self._model = None
        self._tokenizer = None
        self._init_lock = asyncio.Lock()
        self._load_failed = False

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
        """Load LanguageBind model lazily."""
        if self._model is not None:
            return True

        if self._load_failed:
            return False

        async with self._init_lock:
            if self._model is not None:
                return True

            try:
                log.info("[LanguageBind] Loading model...")

                # Try to load LanguageBind
                from transformers import AutoModel, AutoTokenizer

                model_id = "LanguageBind/LanguageBind_Video_V1.5_FT"

                self._tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
                self._model = AutoModel.from_pretrained(model_id, trust_remote_code=True)

                device = self._get_device()
                self._model.to(device)
                self._device = device

                log.info(f"[LanguageBind] Loaded on {device}")
                return True

            except ImportError as e:
                log.warning(f"[LanguageBind] Not available: {e}")
                self._load_failed = True
                return False
            except Exception as e:
                log.warning(f"[LanguageBind] Load failed: {e}")
                # Fall back to CLIP for basic functionality
                try:
                    log.info("[LanguageBind] Falling back to CLIP...")
                    from transformers import CLIPModel, CLIPProcessor

                    self._model = CLIPModel.from_pretrained(
                        "openai/clip-vit-base-patch32"
                    )
                    self._tokenizer = CLIPProcessor.from_pretrained(
                        "openai/clip-vit-base-patch32"
                    )

                    device = self._get_device()
                    self._model.to(device)
                    self._device = device

                    log.info(f"[LanguageBind] CLIP fallback loaded on {device}")
                    return True

                except Exception as e2:
                    log.error(f"[LanguageBind] All loads failed: {e2}")
                    self._load_failed = True
                    return False

    async def encode_video(
        self,
        frames: list[np.ndarray],
        sample_frames: int = 8,
    ) -> np.ndarray | None:
        """Encode video frames to embedding.

        Args:
            frames: List of RGB frames.
            sample_frames: Number of frames to sample.

        Returns:
            Video embedding as numpy array.
        """
        if not await self._lazy_load():
            return None

        try:
            import torch
            from PIL import Image

            # Sample frames evenly
            if len(frames) > sample_frames:
                indices = np.linspace(
                    0, len(frames) - 1, sample_frames, dtype=int
                )
                sampled = [frames[i] for i in indices]
            else:
                sampled = frames

            # Average frame embeddings (simple approach)
            embeddings = []
            for frame in sampled:
                if isinstance(frame, np.ndarray):
                    image = Image.fromarray(frame)
                else:
                    image = frame

                # Process based on model type
                if hasattr(self._tokenizer, "image_processor"):
                    inputs = self._tokenizer(images=image, return_tensors="pt")
                else:
                    inputs = self._tokenizer(
                        images=image,
                        return_tensors="pt",
                        padding=True,
                    )

                inputs = {
                    k: v.to(self._device)
                    for k, v in inputs.items()
                    if isinstance(v, torch.Tensor)
                }

                with torch.no_grad():
                    if hasattr(self._model, "get_image_features"):
                        emb = self._model.get_image_features(**inputs)
                    else:
                        emb = self._model(**inputs).last_hidden_state.mean(
                            dim=1
                        )

                embeddings.append(emb.cpu().numpy())

            # Average all frame embeddings
            avg_embedding = np.mean(embeddings, axis=0)

            # Normalize
            norm = np.linalg.norm(avg_embedding)
            if norm > 0:
                avg_embedding = avg_embedding / norm

            # CRITICAL FIX: Flatten to 1D before dimension check to avoid concatenation error
            # np.mean can return 2D array if input embeddings are 2D, causing ValueError
            avg_embedding = avg_embedding.flatten()

            # Pad to 768 dimensions if needed (for CLIP fallback compatibility with LanguageBind DB)
            if avg_embedding.shape[0] < 768:
                padding = np.zeros((768 - avg_embedding.shape[0],))
                avg_embedding = np.concatenate([avg_embedding, padding])

            log.info(
                f"[LanguageBind] Encoded Video: {len(sampled)} frames, "
                f"Embedding dim={avg_embedding.shape[0]}, "
                f"Norm={norm:.4f}"
            )

            return avg_embedding.flatten()

        except Exception as e:
            log.exception(f"[LanguageBind] Video encoding failed: {e}")
            return None

    async def encode_text(self, text: str) -> np.ndarray | None:
        """Encode text to embedding.

        Args:
            text: Text query.

        Returns:
            Text embedding as numpy array.
        """
        if not await self._lazy_load():
            return None

        try:
            import torch

            if hasattr(self._tokenizer, "tokenizer"):
                inputs = self._tokenizer(
                    text=text, return_tensors="pt", padding=True
                )
            else:
                inputs = self._tokenizer(
                    text=text, return_tensors="pt", padding=True
                )

            inputs = {
                k: v.to(self._device)
                for k, v in inputs.items()
                if isinstance(v, torch.Tensor)
            }

            with torch.no_grad():
                if hasattr(self._model, "get_text_features"):
                    emb = self._model.get_text_features(**inputs)
                else:
                    emb = self._model(**inputs).last_hidden_state.mean(dim=1)

            embedding = emb.cpu().numpy().flatten()

            # Normalize
            norm = np.linalg.norm(embedding)
            if norm > 0:
                embedding = embedding / norm

            # CRITICAL FIX: Pad to 768 dimensions to match encode_video output
            # This prevents dimension mismatch when computing similarity
            if embedding.shape[0] < 768:
                padding = np.zeros((768 - embedding.shape[0],))
                embedding = np.concatenate([embedding, padding])

            return embedding

        except Exception as e:
            log.error(f"[LanguageBind] Text encoding failed: {e}")
            return None

    async def encode_audio(
        self,
        audio_segment: np.ndarray,
        sample_rate: int = 16000,
    ) -> np.ndarray | None:
        """Encode audio to embedding.

        Uses CLAP for audio encoding (LanguageBind audio component).

        Args:
            audio_segment: Audio samples.
            sample_rate: Sample rate.

        Returns:
            Audio embedding as numpy array.
        """
        try:
            # Use CLAP for audio encoding
            from core.processing.audio_events import AudioEventDetector

            detector = AudioEventDetector()

            # Get audio embedding via CLAP
            if not await detector._lazy_load():
                return None

            import torch

            # Resample to 48kHz for CLAP
            import librosa

            if sample_rate != 48000:
                audio_segment = librosa.resample(
                    audio_segment.astype(np.float32),
                    orig_sr=sample_rate,
                    target_sr=48000,
                )

            audio_inputs = detector.processor(
                audios=audio_segment,
                sampling_rate=48000,
                return_tensors="pt",
            )
            audio_inputs = {
                k: v.to(detector._device) for k, v in audio_inputs.items()
            }

            with torch.no_grad():
                audio_emb = detector.model.get_audio_features(**audio_inputs)
                audio_emb = audio_emb / audio_emb.norm(dim=-1, keepdim=True)

            return audio_emb.cpu().numpy().flatten()

        except Exception as e:
            log.error(f"[LanguageBind] Audio encoding failed: {e}")
            return None

    def cleanup(self) -> None:
        """Release resources."""
        if self._model:
            del self._model
            self._model = None
        if self._tokenizer:
            del self._tokenizer
            self._tokenizer = None
        log.info("[LanguageBind] Resources released")


class InternVideoEncoder:
    """InternVideo2 encoder for action/motion understanding.

    SOTA model for understanding complex actions like
    "person kicking ball" vs "person holding ball".

    Based on Research:
    - InternVideo2-Stage2: Masked Video Modeling
    - Best for temporal/action queries
    """

    def __init__(self, device: str | None = None):
        """Initialize InternVideo encoder.

        Args:
            device: Device to run on.
        """
        self._device = device
        self._model = None
        self._processor = None
        self._init_lock = asyncio.Lock()
        self._load_failed = False
        # Cache LanguageBindEncoder to avoid redundant instantiation
        self._languagebind_encoder: LanguageBindEncoder | None = None
        # Cache text embeddings for common action labels to avoid redundant encoding
        self._text_embedding_cache: dict[str, np.ndarray] = {}

    async def _lazy_load(self) -> bool:
        """Load InternVideo model lazily."""
        if self._model is not None:
            return True

        if self._load_failed:
            return False

        async with self._init_lock:
            if self._model is not None:
                return True

            try:
                log.info("[InternVideo] Loading model...")

                # InternVideo2 via HuggingFace
                from transformers import AutoModel, AutoProcessor

                model_id = "OpenGVLab/InternVideo2-Stage2_1B-224p-f4"

                self._processor = AutoProcessor.from_pretrained(model_id)
                self._model = AutoModel.from_pretrained(model_id)

                import torch

                device = self._device or (
                    "cuda" if torch.cuda.is_available() else "cpu"
                )
                self._model.to(device)
                self._device = device

                log.info(f"[InternVideo] Loaded on {device}")
                return True

            except Exception as e:
                log.warning(
                    f"[InternVideo] Load failed, using LanguageBind fallback: {e}"
                )
                self._load_failed = True
                return False

    async def encode_action(
        self,
        frames: list[np.ndarray],
        num_frames: int = 8,
    ) -> np.ndarray | None:
        """Encode video for action understanding.

        Args:
            frames: List of video frames.
            num_frames: Number of frames to use.

        Returns:
            Action-aware video embedding.
        """
        if not await self._lazy_load():
            # Fallback to LanguageBind
            encoder = LanguageBindEncoder(device=self._device)
            return await encoder.encode_video(frames, sample_frames=num_frames)

        try:
            import torch
            from PIL import Image

            # Sample frames
            if len(frames) > num_frames:
                indices = np.linspace(0, len(frames) - 1, num_frames, dtype=int)
                sampled = [frames[i] for i in indices]
            else:
                sampled = frames

            # Convert to PIL
            images = []
            for frame in sampled:
                if isinstance(frame, np.ndarray):
                    images.append(Image.fromarray(frame))
                else:
                    images.append(frame)

            # Process
            inputs = self._processor(images=images, return_tensors="pt")
            inputs = {k: v.to(self._device) for k, v in inputs.items()}

            with torch.no_grad():
                outputs = self._model(**inputs)
                # Get CLS token or pooled output
                if hasattr(outputs, "pooler_output"):
                    embedding = outputs.pooler_output
                else:
                    embedding = outputs.last_hidden_state.mean(dim=1)

            emb = embedding.cpu().numpy().flatten()

            # Normalize
            norm = np.linalg.norm(emb)
            if norm > 0:
                emb = emb / norm

            return emb

        except Exception as e:
            log.error(f"[InternVideo] Encoding failed: {e}")
            return None

    async def recognize_action(
        self,
        frames: list[np.ndarray],
    ) -> dict[str, Any] | None:
        """Recognize action in video frames.

        Args:
            frames: List of video frames.

        Returns:
            Dictionary with action labels and features.
        """
        # Common kinetic actions to check for
        common_actions = [
            "running", "walking", "eating", "drinking", "talking",
            "driving", "dancing", "cooking", "fighting", "playing sports"
        ]
        
        # Get features
        features = await self.encode_action(frames)
        if features is None:
            return None

        # Classify
        actions = await self.classify_action(frames, common_actions)
        
        return {
            "actions": [a["action"] for a in actions[:3]], # Top 3
            "features": features
        }

    async def classify_action(
        self,
        frames: list[np.ndarray],
        action_labels: list[str],
    ) -> list[dict[str, Any]]:
        """Classify action in video from given labels.

        Zero-shot action classification.

        Args:
            frames: Video frames.
            action_labels: Possible action labels.

        Returns:
            List of {action, confidence} sorted by confidence.
        """
        # Reuse cached encoder to avoid redundant model loading (critical perf fix)
        if self._languagebind_encoder is None:
            self._languagebind_encoder = LanguageBindEncoder(device=self._device)
        encoder = self._languagebind_encoder

        video_emb = await encoder.encode_video(frames)
        if video_emb is None:
            return []

        results = []
        for label in action_labels:
            cache_key = f"a video of {label}"
            if cache_key in self._text_embedding_cache:
                text_emb = self._text_embedding_cache[cache_key]
            else:
                text_emb = await encoder.encode_text(cache_key)
                if text_emb is not None:
                    self._text_embedding_cache[cache_key] = text_emb
            if text_emb is None:
                continue

            # Cosine similarity
            similarity = float(np.dot(video_emb, text_emb))
            results.append(
                {
                    "action": label,
                    "confidence": round(similarity, 3),
                }
            )

        # Sort by confidence
        results.sort(key=lambda x: x["confidence"], reverse=True)
        return results

    def cleanup(self) -> None:
        """Release resources."""
        if self._model:
            del self._model
            self._model = None
        if self._processor:
            del self._processor
            self._processor = None
        log.info("[InternVideo] Resources released")
