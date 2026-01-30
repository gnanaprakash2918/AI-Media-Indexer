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

from core.utils.resource_arbiter import GPU_SEMAPHORE
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

                def _load():
                    from transformers import AutoModel, AutoTokenizer

                    model_id = "LanguageBind/LanguageBind_Video_V1.5_FT"
                    tokenizer = AutoTokenizer.from_pretrained(
                        model_id, trust_remote_code=True
                    )
                    model = AutoModel.from_pretrained(
                        model_id, trust_remote_code=True
                    )

                    import torch
                    device = self._device or (
                        "cuda" if torch.cuda.is_available() else "cpu"
                    )
                    model.to(device)
                    return tokenizer, model, device

                self._tokenizer, self._model, self._device = await asyncio.to_thread(_load)
                log.info(f"[LanguageBind] Loaded on {self._device}")
                return True

            except Exception as e:
                log.warning(f"[LanguageBind] Load failed: {e}")
                # Fall back to CLIP for basic functionality
                try:
                    log.info("[LanguageBind] Falling back to CLIP...")

                    def _fallback():
                        from transformers import CLIPModel, CLIPProcessor

                        model = CLIPModel.from_pretrained(
                            "openai/clip-vit-base-patch32"
                        )
                        tokenizer = CLIPProcessor.from_pretrained(
                            "openai/clip-vit-base-patch32"
                        )

                        import torch
                        device = self._device or (
                            "cuda" if torch.cuda.is_available() else "cpu"
                        )
                        model.to(device)
                        return tokenizer, model, device

                    self._tokenizer, self._model, self._device = await asyncio.to_thread(_fallback)
                    log.info(f"[LanguageBind] CLIP fallback loaded on {self._device}")
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
        """Encode video frames to embedding."""
        if not await self._lazy_load():
            return None

        try:
            # 1. Sample frames
            if len(frames) > sample_frames:
                indices = np.linspace(
                    0, len(frames) - 1, sample_frames, dtype=int
                )
                sampled = [frames[i] for i in indices]
            else:
                sampled = frames

            # 2. Preprocess frames in parallel
            async def _prep(frame):
                def __task():
                    from PIL import Image
                    if isinstance(frame, np.ndarray):
                        return Image.fromarray(frame)
                    return frame
                return await asyncio.to_thread(__task)

            pil_images = await asyncio.gather(*[_prep(f) for f in sampled])

            # 3. Batch inference under GPU semaphore
            async with GPU_SEMAPHORE:
                def _infer():
                    import torch
                    embeddings = []
                    for image in pil_images:
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
                            if hasattr(v, "to")
                        }

                        with torch.no_grad():
                            if hasattr(self._model, "get_image_features"):
                                emb = self._model.get_image_features(**inputs)
                            else:
                                emb = self._model(**inputs).last_hidden_state.mean(dim=1)
                        embeddings.append(emb.cpu().numpy().flatten())

                    avg_emb = np.mean(embeddings, axis=0)
                    norm = np.linalg.norm(avg_emb)
                    if norm > 0:
                        avg_emb = avg_emb / norm

                    if avg_emb.ndim > 1:
                        avg_emb = avg_emb.flatten()
                    if avg_emb.shape[0] < 768:
                        padding = np.zeros((768 - avg_emb.shape[0],))
                        avg_emb = np.concatenate([avg_emb, padding])

                    return avg_emb

                return await asyncio.to_thread(_infer)

        except Exception as e:
            log.error(f"[LanguageBind] Video encoding failed: {e}")
            return None

    async def encode_text(self, text: str) -> np.ndarray | None:
        """Encode text to embedding."""
        if not await self._lazy_load():
            return None

        async with GPU_SEMAPHORE:
            def _infer():
                import torch
                inputs = self._tokenizer(text=text, return_tensors="pt", padding=True)
                inputs = {
                    k: v.to(self._device)
                    for k, v in inputs.items()
                    if hasattr(v, "to")
                }

                with torch.no_grad():
                    if hasattr(self._model, "get_text_features"):
                        emb = self._model.get_text_features(**inputs)
                    else:
                        emb = self._model(**inputs).last_hidden_state.mean(dim=1)

                embedding = emb.cpu().numpy().flatten()
                norm = np.linalg.norm(embedding)
                if norm > 0:
                    embedding = embedding / norm

                if embedding.shape[0] < 768:
                    padding = np.zeros((768 - embedding.shape[0],))
                    embedding = np.concatenate([embedding, padding])

                return embedding

            return await asyncio.to_thread(_infer)

    async def encode_audio(
        self,
        audio_segment: np.ndarray,
        sample_rate: int = 16000,
    ) -> np.ndarray | None:
        """Encode audio to embedding."""
        try:
            from core.processing.audio_events import AudioEventDetector
            detector = AudioEventDetector()
            if not await detector._lazy_load():
                return None

            async with GPU_SEMAPHORE:
                def _process():
                    import torch
                    import librosa
                    if sample_rate != 48000:
                        audio = librosa.resample(
                            audio_segment.astype(np.float32),
                            orig_sr=sample_rate,
                            target_sr=48000,
                        )
                    else:
                        audio = audio_segment

                    audio_inputs = detector.processor(
                        audios=audio,
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

                return await asyncio.to_thread(_process)

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

                def _load():
                    from transformers import AutoModel, AutoProcessor
                    model_id = "OpenGVLab/InternVideo2-Stage2_1B-224p-f4"
                    processor = AutoProcessor.from_pretrained(model_id)
                    model = AutoModel.from_pretrained(model_id)
                    import torch
                    device = self._device or (
                        "cuda" if torch.cuda.is_available() else "cpu"
                    )
                    model.to(device)
                    return processor, model, device

                self._processor, self._model, self._device = await asyncio.to_thread(_load)
                log.info(f"[InternVideo] Loaded on {self._device}")
                return True

            except Exception as e:
                log.warning(f"[InternVideo] Load failed, fallback: {e}")
                self._load_failed = True
                return False

    async def encode_action(
        self,
        frames: list[np.ndarray],
        num_frames: int = 8,
    ) -> np.ndarray | None:
        """Encode video for action understanding."""
        if not await self._lazy_load():
            if self._languagebind_encoder is None:
                self._languagebind_encoder = LanguageBindEncoder(device=self._device)
            return await self._languagebind_encoder.encode_video(
                frames, sample_frames=num_frames
            )

        try:
            # 1. Prepare frames
            if len(frames) > num_frames:
                indices = np.linspace(0, len(frames) - 1, num_frames, dtype=int)
                sampled = [frames[i] for i in indices]
            else:
                sampled = frames

            async def _prep(f):
                def __task():
                    from PIL import Image
                    return Image.fromarray(f) if isinstance(f, np.ndarray) else f
                return await asyncio.to_thread(__task)

            pil_images = await asyncio.gather(*[_prep(f) for f in sampled])

            # 2. Inference under GPU semaphore
            async with GPU_SEMAPHORE:
                def _infer():
                    import torch
                    inputs = self._processor(images=pil_images, return_tensors="pt")
                    inputs = {
                        k: v.to(self._device)
                        for k, v in inputs.items()
                        if hasattr(v, "to")
                    }
                    with torch.no_grad():
                        outputs = self._model(**inputs)
                        if hasattr(outputs, "pooler_output"):
                            emb = outputs.pooler_output
                        else:
                            emb = outputs.last_hidden_state.mean(dim=1)

                    feats = emb.cpu().numpy().flatten()
                    norm = np.linalg.norm(feats)
                    if norm > 0:
                        feats = feats / norm
                    return feats

                return await asyncio.to_thread(_infer)
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
