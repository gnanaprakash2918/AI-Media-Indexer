import asyncio
import io
import logging
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

from core.utils.resource_arbiter import GPU_SEMAPHORE

log = logging.getLogger(__name__)


class VisualEncoderInterface(ABC):
    """Abstract base class for visual encoders."""

    @property
    @abstractmethod
    def embedding_dim(self) -> int: ...

    @property
    @abstractmethod
    def name(self) -> str: ...

    @abstractmethod
    async def encode_image(
        self, image: np.ndarray | bytes | Path
    ) -> list[float]: ...

    @abstractmethod
    async def encode_batch(
        self, images: list[np.ndarray | bytes | Path]
    ) -> list[list[float]]: ...

    @abstractmethod
    async def encode_text(self, text: str) -> list[float]: ...

    def cleanup(self) -> None: ...



class BaseVisualEncoder(VisualEncoderInterface, ABC):
    """Shared base class for all visual encoders to eliminate duplicate code."""

    def __init__(self, model_name: str) -> None:
        self._model_name = model_name
        self._model = None
        self._preprocess = None
        self._tokenizer = None
        self._load_lock = asyncio.Lock()
        
    @property
    def name(self) -> str:
        return self.__class__.__name__.replace("Encoder", "")

    @abstractmethod
    def _load(self) -> None:
        """Subclass implementation of actual model loading."""
        pass
        
    async def _lazy_init(self) -> None:
        """Thread-safe lazy initialization of the model."""
        if self._model is not None:
            return

        async with self._load_lock:
            # Double check locking pattern
            if self._model is not None:
                return
            
            try:
                # Run the actual load in a thread pool to avoid blocking the event loop
                await asyncio.to_thread(self._load)
            except Exception as e:
                log.error(f"Failed to load {self.name} model: {e}")
                raise

    @abstractmethod
    def _process_image_internal(self, image: np.ndarray | bytes | Path) -> Any:
        """Subclass implementation of image preprocessing."""
        pass
        
    @abstractmethod
    def _infer_image_internal(self, inputs: Any) -> np.ndarray:
        """Subclass implementation of image inference."""
        pass

    async def encode_image(self, image: np.ndarray | bytes | Path) -> np.ndarray:
        """Extract visual embedding from a single image."""
        await self._lazy_init()
        
        try:
            # Preprocess in thread pool
            inputs = await asyncio.to_thread(self._process_image_internal, image)
            if inputs is None:
                return np.zeros(self.embedding_dim)

            # Infer (runs on GPU, so acquire semaphore)
            async with GPU_SEMAPHORE:
                embedding = await asyncio.to_thread(self._infer_image_internal, inputs)
                
            return embedding
        except Exception as e:
            log.error(f"Error encoding image with {self.name}: {e}")
            return np.zeros(self.embedding_dim)

    @abstractmethod
    def _infer_batch_internal(self, inputs: list[Any]) -> list[np.ndarray]:
        """Subclass implementation of batch image inference."""
        pass

    async def encode_batch(self, images: list[np.ndarray | bytes | Path]) -> list[np.ndarray]:
        """Extract visual embeddings from a batch of images."""
        if not images:
            return []

        await self._lazy_init()
        
        try:
            # Preprocess all images in parallel
            tasks = [asyncio.to_thread(self._process_image_internal, img) for img in images]
            processed_inputs = await asyncio.gather(*tasks)
            
            # Filter out failures
            valid_inputs = [inp for inp in processed_inputs if inp is not None]
            
            if not valid_inputs:
                return [np.zeros(self.embedding_dim) for _ in images]

            # Infer batch
            async with GPU_SEMAPHORE:
                embeddings = await asyncio.to_thread(self._infer_batch_internal, valid_inputs)

            # Map results back, filling zeros for failures
            result = []
            emb_idx = 0
            for inp in processed_inputs:
                if inp is None:
                    result.append(np.zeros(self.embedding_dim))
                else:
                    if emb_idx < len(embeddings):
                        result.append(embeddings[emb_idx])
                    else:
                        result.append(np.zeros(self.embedding_dim))
                    emb_idx += 1
                    
            return result
        except Exception as e:
            log.error(f"Error encoding batch with {self.name}: {e}")
            return [np.zeros(self.embedding_dim) for _ in images]

    @abstractmethod
    def _infer_text_internal(self, text: str) -> np.ndarray:
        """Subclass implementation of text inference."""
        pass

    async def encode_text(self, text: str) -> np.ndarray:
        """Extract embedding from text for search semantic alignment."""
        if not text.strip():
            return np.zeros(self.embedding_dim)

        await self._lazy_init()
        
        try:
            async with GPU_SEMAPHORE:
                embedding = await asyncio.to_thread(self._infer_text_internal, text)
            return embedding
        except Exception as e:
            log.error(f"Error encoding text with {self.name}: {e}")
            return np.zeros(self.embedding_dim)

    def cleanup(self) -> None:
        """Free GPU resources aggressively."""
        if hasattr(self, "_model") and self._model is not None:
            del self._model
            self._model = None
            
        if hasattr(self, "_preprocess") and self._preprocess is not None:
            del self._preprocess
            self._preprocess = None
            
        if hasattr(self, "_tokenizer") and self._tokenizer is not None:
            del self._tokenizer
            self._tokenizer = None
            
        try:
            import torch
            import gc
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            gc.collect()
        except ImportError:
            pass


class CLIPEncoder(BaseVisualEncoder):
    """OpenAI CLIP visual encoder with async support."""

    def __init__(self, model_name: str = "ViT-L-14", pretrained: str = "openai"):
        super().__init__(model_name)
        self._pretrained = pretrained

    @property
    def embedding_dim(self) -> int:
        return 768  # For ViT-L-14

    def _load(self) -> None:
        try:
            import torch
            import open_clip

            device = "cuda" if torch.cuda.is_available() else "cpu"
            model, _, preprocess = open_clip.create_model_and_transforms(
                self._model_name,
                pretrained=self._pretrained,
                device=device,
            )
            tokenizer = open_clip.get_tokenizer(self._model_name)

            self._model = model
            self._preprocess = preprocess
            self._tokenizer = tokenizer
            self._device = device
            
            # Optimize for inference
            self._model.eval()
            if device == "cuda":
                # Use half precision for speed if supported
                pass
        except ImportError:
            raise RuntimeError("open_clip module is required for CLIPEncoder")

    def _process_image_internal(self, image: np.ndarray | bytes | Path) -> Any:
        try:
            if isinstance(image, bytes):
                pil_img = Image.open(io.BytesIO(image)).convert("RGB")
            elif isinstance(image, Path) or isinstance(image, str):
                pil_img = Image.open(image).convert("RGB")
            elif isinstance(image, np.ndarray):
                # Ensure correct format for PIL
                if image.dtype != np.uint8:
                    if image.max() <= 1.0:
                        image = (image * 255).astype(np.uint8)
                    else:
                        image = image.astype(np.uint8)
                pil_img = Image.fromarray(image).convert("RGB")
            else:
                return None
                
            return self._preprocess(pil_img).unsqueeze(0).to(self._device)
        except Exception:
            return None

    def _infer_image_internal(self, inputs: Any) -> np.ndarray:
        import torch
        with torch.no_grad():
            image_features = self._model.encode_image(inputs)
            # Normalize
            image_features /= image_features.norm(dim=-1, keepdim=True)
            return image_features.cpu().numpy().flatten()

    def _infer_batch_internal(self, inputs: list[Any]) -> list[np.ndarray]:
        import torch
        with torch.no_grad():
            # Already passed as a list of [1, C, H, W] tensors
            # Need to concatenate them along dim 0
            batch_tensor = torch.cat(inputs, dim=0)
            
            image_features = self._model.encode_image(batch_tensor)
            image_features /= image_features.norm(dim=-1, keepdim=True)
            
            embeddings = image_features.cpu().numpy()
            return [emb for emb in embeddings]

    def _infer_text_internal(self, text: str) -> np.ndarray:
        import torch
        with torch.no_grad():
            text_tokens = self._tokenizer([text]).to(self._device)
            text_features = self._model.encode_text(text_tokens)
            # Normalize
            text_features /= text_features.norm(dim=-1, keepdim=True)
            return text_features.cpu().numpy().flatten()


class SigLIPEncoder(BaseVisualEncoder):
    """Google SigLIP visual encoder (SOTA) with async support."""

    def __init__(self, model_name: str = "ViT-SO400M-14-SigLIP-384"):
        super().__init__(model_name)
        # Using the HuggingFace webli variant as default
        self._hf_model_id = "google/siglip-so400m-patch14-384"
        
        # Override to smaller model if environment doesn't have much VRAM
        try:
            import torch
            if torch.cuda.is_available():
                vram_gb = torch.cuda.get_device_properties(0).total_memory / (1024**3)
                if vram_gb < 10.0:
                    # Fallback to smaller model for low VRAM
                    self._hf_model_id = "google/siglip-base-patch16-256"
                    log.warning(f"Low VRAM detected ({vram_gb:.1f}GB). Using smaller SigLIP model: {self._hf_model_id}")
        except:
            pass

    @property
    def embedding_dim(self) -> int:
        # 1152 for SO400M, 768 for base
        return 1152 if "so400m" in self._hf_model_id.lower() else 768

    def _load(self) -> None:
        try:
            import torch
            from transformers import AutoModel, AutoProcessor

            device = "cuda" if torch.cuda.is_available() else "cpu"
            
            # Load with appropriate precision for speed and VRAM savings
            torch_dtype = torch.float16 if device == "cuda" else torch.float32
            
            self._model = AutoModel.from_pretrained(
                self._hf_model_id, 
                torch_dtype=torch_dtype,
                low_cpu_mem_usage=True
            ).to(device)
            self._model.eval()
            
            self._processor = AutoProcessor.from_pretrained(self._hf_model_id)
            self._device = device
            
            # Create dummy functions to match BaseVisualEncoder expectations
            self._preprocess = self._processor
            self._tokenizer = self._processor
        except ImportError:
            raise RuntimeError("transformers module is required for SigLIPEncoder")
        except Exception as e:
            log.error(f"Failed to load SigLIP model {self._hf_model_id}: {e}")
            raise

    def _process_image_internal(self, image: np.ndarray | bytes | Path) -> Any:
        try:
            if isinstance(image, bytes):
                pil_img = Image.open(io.BytesIO(image)).convert("RGB")
            elif isinstance(image, Path) or isinstance(image, str):
                pil_img = Image.open(image).convert("RGB")
            elif isinstance(image, np.ndarray):
                if image.dtype != np.uint8:
                    image = (image * 255).astype(np.uint8) if image.max() <= 1.0 else image.astype(np.uint8)
                pil_img = Image.fromarray(image).convert("RGB")
            else:
                return None
                
            # SigLIP processor returns a dict with pixel_values tensor
            inputs = self._processor(images=pil_img, return_tensors="pt")
            
            # Map to expected device and dtype
            if self._device == "cuda":
                import torch
                inputs = {k: v.to(self._device, dtype=torch.float16 if v.dtype == torch.float32 else v.dtype) 
                          for k, v in inputs.items()}
            else:
                inputs = {k: v.to(self._device) for k, v in inputs.items()}
                
            return inputs
        except Exception:
            return None

    def _infer_image_internal(self, inputs: Any) -> np.ndarray:
        import torch
        with torch.no_grad():
            outputs = self._model.get_image_features(**inputs)
            # Normalize
            outputs /= outputs.norm(dim=-1, keepdim=True)
            return outputs.cpu().float().numpy().flatten()

    def _infer_batch_internal(self, inputs: list[Any]) -> list[np.ndarray]:
        import torch
        with torch.no_grad():
            # Combine individual dict inputs into a single batch
            batch_inputs = {}
            for k in inputs[0].keys():
                batch_inputs[k] = torch.cat([inp[k] for inp in inputs], dim=0)
                
            outputs = self._model.get_image_features(**batch_inputs)
            outputs /= outputs.norm(dim=-1, keepdim=True)
            
            embeddings = outputs.cpu().float().numpy()
            return [emb for emb in embeddings]

    def _infer_text_internal(self, text: str) -> np.ndarray:
        import torch
        with torch.no_grad():
            inputs = self._processor(text=[text], padding="max_length", return_tensors="pt")
            inputs = {k: v.to(self._device) for k, v in inputs.items()}
            
            outputs = self._model.get_text_features(**inputs)
            # Normalize
            outputs /= outputs.norm(dim=-1, keepdim=True)
            return outputs.cpu().float().numpy().flatten()



