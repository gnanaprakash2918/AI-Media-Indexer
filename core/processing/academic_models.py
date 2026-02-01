"""Academic SOTA Models: V-JEPA, DINOv2, VideoMAE, ImageBind.

The "Godfather Architectures" - cutting-edge self-supervised models
for advanced video understanding.

Based on Research (Part 4):
- I-JEPA / V-JEPA (Meta/LeCun): Motion Prediction
- DINOv2 (Meta): Zero-shot Object Discovery
- VideoMAE V2: Action Recognition
- ImageBind (Meta): 6-modality unified space
"""

from __future__ import annotations

import asyncio

import numpy as np

from core.utils.logger import get_logger

log = get_logger(__name__)


class VJEPAEncoder:
    """V-JEPA (Video Joint-Embedding Predictive Architecture).

    Yann LeCun's architecture that predicts representations
    of missing video parts - superior for motion understanding.

    Based on Research:
    - Predicts *representation* not pixels
    - Best for "what happens next" queries

    Usage:
        encoder = VJEPAEncoder()
        motion_features = await encoder.extract_motion_features(frames)
    """

    def __init__(self, device: str | None = None):
        """Initialize V-JEPA encoder.

        Args:
            device: Device to run on.
        """
        self._device = device
        self._model = None
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
        """Load V-JEPA model lazily."""
        if self._model is not None:
            return True

        if self._load_failed:
            return False

        async with self._init_lock:
            if self._model is not None:
                return True

            try:
                log.info("[V-JEPA] Loading model...")

                # V-JEPA from Meta's official release
                import torch

                # Try to load from torch hub
                self._model = torch.hub.load(
                    "facebookresearch/jepa",
                    "vjepa_huge_patch14_224",
                    pretrained=True,
                )

                device = self._get_device()
                self._model.to(device)
                self._model.eval()
                self._device = device

                log.info(f"[V-JEPA] Loaded on {device}")
                return True

            except Exception as e:
                log.warning(f"[V-JEPA] Official model not available: {e}")
                log.info(
                    "[V-JEPA] Using TimeSformer fallback for motion features..."
                )

                try:
                    # Fallback to TimeSformer which also captures temporal dynamics
                    from transformers import TimesformerModel

                    self._model = TimesformerModel.from_pretrained(
                        "facebook/timesformer-base-finetuned-k400"
                    )

                    device = self._get_device()
                    self._model.to(device)
                    self._model.eval()
                    self._device = device

                    log.info(
                        f"[V-JEPA] TimeSformer fallback loaded on {device}"
                    )
                    return True

                except Exception as e2:
                    log.error(f"[V-JEPA] All loads failed: {e2}")
                    self._load_failed = True
                    return False

    async def extract_motion_features(
        self,
        frames: list[np.ndarray],
        num_frames: int = 8,
    ) -> np.ndarray | None:
        """Extract motion-aware features from video frames.

        V-JEPA excels at understanding temporal dynamics
        and "what happens next" predictions.

        Args:
            frames: List of RGB video frames.
            num_frames: Number of frames to process.

        Returns:
            Motion-aware feature embedding.
        """
        if not await self._lazy_load():
            return None

        try:
            import torch
            from PIL import Image
            from torchvision import transforms

            # Sample frames evenly
            if len(frames) > num_frames:
                indices = np.linspace(0, len(frames) - 1, num_frames, dtype=int)
                sampled = [frames[i] for i in indices]
            else:
                sampled = frames
                # Pad if needed
                while len(sampled) < num_frames:
                    sampled.append(sampled[-1])

            # Preprocess
            transform = transforms.Compose(
                [
                    transforms.Resize(224),
                    transforms.CenterCrop(224),
                    transforms.ToTensor(),
                    transforms.Normalize(
                        mean=[0.485, 0.456, 0.406],
                        std=[0.229, 0.224, 0.225],
                    ),
                ]
            )

            processed = []
            for frame in sampled:
                if isinstance(frame, np.ndarray):
                    image = Image.fromarray(frame)
                else:
                    image = frame
                processed.append(transform(image))

            # Stack: [num_frames, C, H, W] -> [1, num_frames, C, H, W]
            video_tensor = torch.stack(processed).unsqueeze(0).to(self._device)

            with torch.no_grad():
                # TimeSformer expects [B, C, T, H, W]
                video_tensor = video_tensor.permute(0, 2, 1, 3, 4)
                outputs = self._model(video_tensor)

                if hasattr(outputs, "last_hidden_state"):
                    features = outputs.last_hidden_state.mean(dim=1)
                else:
                    features = outputs

            embedding = features.cpu().numpy().flatten()

            # Normalize
            norm = np.linalg.norm(embedding)
            if norm > 0:
                embedding = embedding / norm

            return embedding

        except Exception as e:
            log.error(f"[V-JEPA] Feature extraction failed: {e}")
            return None

    async def predict_next_frame_features(
        self,
        frames: list[np.ndarray],
    ) -> np.ndarray | None:
        """Predict features of what comes next in the video.

        This is V-JEPA's core capability - predicting future states
        without generating pixels.

        Args:
            frames: Context frames.

        Returns:
            Predicted feature embedding for next frame.
        """
        # V-JEPA's actual prediction requires the trained decoder
        # For now, use motion features as proxy
        return await self.extract_motion_features(frames)

    def cleanup(self) -> None:
        """Release resources."""
        if self._model:
            del self._model
            self._model = None
        log.info("[V-JEPA] Resources released")


class DINOv2Encoder:
    """DINOv2 (Self-supervised Vision Transformer).

    Meta's self-supervised model with unmatched dense features.
    Understands object parts (legs, wheels) without labels.

    Based on Research:
    - Best for Zero-shot Object Discovery
    - Dense features for segmentation

    Usage:
        encoder = DINOv2Encoder()
        features = await encoder.extract_features(frame)
        parts = await encoder.discover_object_parts(frame)
    """

    def __init__(self, model_size: str = "base", device: str | None = None):
        """Initialize DINOv2 encoder.

        Args:
            model_size: Model size ('small', 'base', 'large', 'giant').
            device: Device to run on.
        """
        self.model_size = model_size
        self._device = device
        self._model = None
        self._processor = None
        self._init_lock = asyncio.Lock()

    async def _lazy_load(self) -> bool:
        """Load DINOv2 model lazily."""
        if self._model is not None:
            return True

        async with self._init_lock:
            if self._model is not None:
                return True

            try:
                log.info(f"[DINOv2] Loading {self.model_size} model...")

                import torch

                # Load from torch hub
                model_name = f"dinov2_vit{self.model_size[0]}14"
                self._model = torch.hub.load(
                    "facebookresearch/dinov2",
                    model_name,
                )

                device = self._device or (
                    "cuda" if torch.cuda.is_available() else "cpu"
                )
                self._model.to(device)
                self._model.eval()
                self._device = device

                log.info(f"[DINOv2] Loaded on {device}")
                return True

            except Exception as e:
                log.warning(f"[DINOv2] torch.hub failed: {e}")

                try:
                    # Fallback to HuggingFace
                    from transformers import AutoImageProcessor, AutoModel

                    model_id = f"facebook/dinov2-{self.model_size}"
                    self._processor = AutoImageProcessor.from_pretrained(
                        model_id
                    )
                    self._model = AutoModel.from_pretrained(model_id)

                    import torch

                    device = self._device or (
                        "cuda" if torch.cuda.is_available() else "cpu"
                    )
                    self._model.to(device)
                    self._device = device

                    log.info(f"[DINOv2] HuggingFace model loaded on {device}")
                    return True

                except Exception as e2:
                    log.error(f"[DINOv2] All loads failed: {e2}")
                    return False

    async def extract_features(
        self,
        frame: np.ndarray,
    ) -> np.ndarray | None:
        """Extract DINOv2 features from a frame.

        Args:
            frame: RGB frame.

        Returns:
            Dense feature embedding.
        """
        if not await self._lazy_load():
            return None

        try:
            import torch
            from PIL import Image
            from torchvision import transforms

            if isinstance(frame, np.ndarray):
                image = Image.fromarray(frame)
            else:
                image = frame

            if self._processor:
                # HuggingFace path
                inputs = self._processor(images=image, return_tensors="pt")
                inputs = {k: v.to(self._device) for k, v in inputs.items()}

                with torch.no_grad():
                    outputs = self._model(**inputs)
                    features = outputs.last_hidden_state.mean(dim=1)
            else:
                # torch.hub path
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

                tensor = transform(image).unsqueeze(0).to(self._device)

                with torch.no_grad():
                    features = self._model(tensor)

            embedding = features.cpu().numpy().flatten()

            # Normalize
            norm = np.linalg.norm(embedding)
            if norm > 0:
                embedding = embedding / norm

            return embedding

        except Exception as e:
            log.error(f"[DINOv2] Feature extraction failed: {e}")
            return None

    async def get_patch_features(
        self,
        frame: np.ndarray,
    ) -> np.ndarray | None:
        """Get per-patch features for dense understanding.

        Returns features for each patch, enabling
        part-level object understanding.

        Args:
            frame: RGB frame.

        Returns:
            Patch features of shape [num_patches, feature_dim].
        """
        if not await self._lazy_load():
            return None

        try:
            import torch
            from PIL import Image
            from torchvision import transforms

            if isinstance(frame, np.ndarray):
                image = Image.fromarray(frame)
            else:
                image = frame

            if self._processor:
                inputs = self._processor(images=image, return_tensors="pt")
                inputs = {k: v.to(self._device) for k, v in inputs.items()}

                with torch.no_grad():
                    outputs = self._model(**inputs)
                    # All patch embeddings (excluding CLS token)
                    patch_features = outputs.last_hidden_state[:, 1:, :]
            else:
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

                tensor = transform(image).unsqueeze(0).to(self._device)

                with torch.no_grad():
                    # Get intermediate features
                    patch_features = self._model.get_intermediate_layers(
                        tensor, n=1
                    )[0]

            return patch_features.cpu().numpy().squeeze()

        except Exception as e:
            log.error(f"[DINOv2] Patch feature extraction failed: {e}")
            return None

    def cleanup(self) -> None:
        """Release resources."""
        if self._model:
            del self._model
            self._model = None
        if self._processor:
            del self._processor
            self._processor = None
        log.info("[DINOv2] Resources released")


class VideoMAEEncoder:
    """VideoMAE V2 encoder for action recognition.

    Masks 90% of video tubelets and learns to predict them.
    Best backbone for action recognition tasks.

    Based on Research:
    - Masked Video Modeling
    - SOTA action recognition

    Usage:
        encoder = VideoMAEEncoder()
        action_features = await encoder.extract_action_features(frames)
        actions = await encoder.classify_action(frames, labels)
    """

    def __init__(self, device: str | None = None):
        """Initialize VideoMAE encoder.

        Args:
            device: Device to run on.
        """
        self._device = device
        self._model = None
        self._processor = None
        self._init_lock = asyncio.Lock()

    async def _lazy_load(self) -> bool:
        """Load VideoMAE model lazily."""
        if self._model is not None:
            return True

        async with self._init_lock:
            if self._model is not None:
                return True

            try:
                log.info("[VideoMAE] Loading model...")

                from transformers import VideoMAEImageProcessor, VideoMAEModel

                model_id = "MCG-NJU/videomae-base-finetuned-kinetics"
                self._processor = VideoMAEImageProcessor.from_pretrained(
                    model_id
                )
                self._model = VideoMAEModel.from_pretrained(model_id)

                import torch

                device = self._device or (
                    "cuda" if torch.cuda.is_available() else "cpu"
                )
                self._model.to(device)
                self._device = device

                log.info(f"[VideoMAE] Loaded on {device}")
                return True

            except Exception as e:
                log.error(f"[VideoMAE] Load failed: {e}")
                return False

    async def extract_action_features(
        self,
        frames: list[np.ndarray],
        num_frames: int = 16,
    ) -> np.ndarray | None:
        """Extract action-aware features from video.

        Args:
            frames: List of RGB video frames.
            num_frames: Number of frames to process.

        Returns:
            Action-aware feature embedding.
        """
        if not await self._lazy_load():
            return None

        try:
            import torch
            from PIL import Image

            # Sample frames evenly
            if len(frames) > num_frames:
                indices = np.linspace(0, len(frames) - 1, num_frames, dtype=int)
                sampled = [frames[i] for i in indices]
            else:
                sampled = frames
                while len(sampled) < num_frames:
                    sampled.append(sampled[-1])

            # Convert to PIL
            images = []
            for frame in sampled:
                if isinstance(frame, np.ndarray):
                    images.append(Image.fromarray(frame))
                else:
                    images.append(frame)

            # Process
            inputs = self._processor(images, return_tensors="pt")
            inputs = {k: v.to(self._device) for k, v in inputs.items()}

            with torch.no_grad():
                outputs = self._model(**inputs)
                features = outputs.last_hidden_state.mean(dim=1)

            embedding = features.cpu().numpy().flatten()

            # Normalize
            norm = np.linalg.norm(embedding)
            if norm > 0:
                embedding = embedding / norm

            return embedding

        except Exception as e:
            log.error(f"[VideoMAE] Feature extraction failed: {e}")
            return None

    def cleanup(self) -> None:
        """Release resources."""
        if self._model:
            del self._model
            self._model = None
        if self._processor:
            del self._processor
            self._processor = None
        log.info("[VideoMAE] Resources released")


class ImageBindEncoder:
    """ImageBind: Unified 6-modality embedding space.

    Binds Image, Text, Audio, Depth, Thermal, IMU into one space.
    Enables cross-modal search like "hum a tune -> find video".

    Based on Research:
    - Meta's ImageBind
    - "Search with Audio" capability

    Usage:
        encoder = ImageBindEncoder()
        img_emb = await encoder.encode_image(frame)
        audio_emb = await encoder.encode_audio(audio)
        # Both in same space - can compute similarity!
    """

    def __init__(self, device: str | None = None):
        """Initialize ImageBind encoder.

        Args:
            device: Device to run on.
        """
        self._device = device
        self._model = None
        self._init_lock = asyncio.Lock()
        self._load_failed = False

    async def _lazy_load(self) -> bool:
        """Load ImageBind model lazily."""
        if self._model is not None:
            return True

        if self._load_failed:
            return False

        async with self._init_lock:
            if self._model is not None:
                return True

            try:
                log.info("[ImageBind] Loading model...")

                import torch

                # Try official ImageBind
                self._model = torch.hub.load(
                    "facebookresearch/ImageBind",
                    "imagebind_huge",
                    pretrained=True,
                )

                device = self._device or (
                    "cuda" if torch.cuda.is_available() else "cpu"
                )
                self._model.to(device)
                self._model.eval()
                self._device = device

                log.info(f"[ImageBind] Loaded on {device}")
                return True

            except Exception as e:
                log.warning(f"[ImageBind] Official model not available: {e}")
                log.info("[ImageBind] Using CLIP + CLAP fallback...")

                # Fallback: Use CLIP for vision, CLAP for audio
                self._load_failed = True
                return False

    async def encode_image(
        self,
        frame: np.ndarray,
    ) -> np.ndarray | None:
        """Encode image to ImageBind space.

        Args:
            frame: RGB frame.

        Returns:
            Image embedding in unified space.
        """
        if not await self._lazy_load():
            # Fallback to LanguageBind
            from core.processing.video_understanding import LanguageBindEncoder

            encoder = LanguageBindEncoder(device=self._device)
            return await encoder.encode_video([frame], sample_frames=1)

        try:
            import torch
            from PIL import Image
            from torchvision import transforms

            if isinstance(frame, np.ndarray):
                image = Image.fromarray(frame)
            else:
                image = frame

            # ImageBind preprocessing
            transform = transforms.Compose(
                [
                    transforms.Resize(224),
                    transforms.CenterCrop(224),
                    transforms.ToTensor(),
                    transforms.Normalize(
                        mean=[0.485, 0.456, 0.406],
                        std=[0.229, 0.224, 0.225],
                    ),
                ]
            )

            tensor = transform(image).unsqueeze(0).to(self._device)

            with torch.no_grad():
                embeddings = self._model(
                    {
                        "vision": tensor,
                    }
                )
                embedding = embeddings["vision"]

            emb = embedding.cpu().numpy().flatten()
            norm = np.linalg.norm(emb)
            if norm > 0:
                emb = emb / norm

            return emb

        except Exception as e:
            log.error(f"[ImageBind] Image encoding failed: {e}")
            return None

    async def encode_audio(
        self,
        audio_segment: np.ndarray,
        sample_rate: int = 16000,
    ) -> np.ndarray | None:
        """Encode audio to ImageBind space.

        Enables "hum a tune" or "describe a sound" search.

        Args:
            audio_segment: Audio samples.
            sample_rate: Sample rate.

        Returns:
            Audio embedding in unified space.
        """
        if not await self._lazy_load():
            # Fallback to CLAP
            from core.processing.video_understanding import LanguageBindEncoder

            encoder = LanguageBindEncoder(device=self._device)
            return await encoder.encode_audio(audio_segment, sample_rate)

        try:
            import torch
            import torchaudio

            # ImageBind expects specific audio format
            waveform = torch.from_numpy(audio_segment).float()
            if len(waveform.shape) == 1:
                waveform = waveform.unsqueeze(0)

            # Resample if needed
            if sample_rate != 16000:
                resampler = torchaudio.transforms.Resample(sample_rate, 16000)
                waveform = resampler(waveform)

            waveform = waveform.to(self._device)

            with torch.no_grad():
                embeddings = self._model(
                    {
                        "audio": waveform.unsqueeze(0),
                    }
                )
                embedding = embeddings["audio"]

            emb = embedding.cpu().numpy().flatten()
            norm = np.linalg.norm(emb)
            if norm > 0:
                emb = emb / norm

            return emb

        except Exception as e:
            log.error(f"[ImageBind] Audio encoding failed: {e}")
            return None

    async def encode_text(self, text: str) -> np.ndarray | None:
        """Encode text to ImageBind space.

        Args:
            text: Text query.

        Returns:
            Text embedding in unified space.
        """
        if not await self._lazy_load():
            from core.processing.video_understanding import LanguageBindEncoder

            encoder = LanguageBindEncoder(device=self._device)
            return await encoder.encode_text(text)

        try:
            import torch

            with torch.no_grad():
                embeddings = self._model(
                    {
                        "text": [text],
                    }
                )
                embedding = embeddings["text"]

            emb = embedding.cpu().numpy().flatten()
            norm = np.linalg.norm(emb)
            if norm > 0:
                emb = emb / norm

            return emb

        except Exception as e:
            log.error(f"[ImageBind] Text encoding failed: {e}")
            return None

    async def cross_modal_search(
        self,
        query_embedding: np.ndarray,
        candidate_embeddings: list[tuple[str, np.ndarray]],
        top_k: int = 5,
    ) -> list[tuple[str, float]]:
        """Search across modalities using unified embeddings.

        Query can be audio, image, or text - all searchable
        against candidates from any modality.

        Args:
            query_embedding: Query embedding (any modality).
            candidate_embeddings: List of (id, embedding) tuples.
            top_k: Number of results.

        Returns:
            List of (id, similarity) sorted by score.
        """
        results = []

        for cid, cemb in candidate_embeddings:
            similarity = float(np.dot(query_embedding, cemb))
            results.append((cid, similarity))

        results.sort(key=lambda x: x[1], reverse=True)
        return results[:top_k]

    def cleanup(self) -> None:
        """Release resources."""
        if self._model:
            del self._model
            self._model = None
        log.info("[ImageBind] Resources released")
