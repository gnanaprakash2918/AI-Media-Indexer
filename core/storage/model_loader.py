"""Model loading logic for VectorDB."""
import time
import torch
import shutil
from huggingface_hub import snapshot_download
from sentence_transformers import SentenceTransformer

from config import settings
from core.utils.logger import log

class ModelLoader:
    """Handles loading and managing the SentenceTransformer model."""

    def __init__(self):
        self._model = None
        self._last_used = 0
        self._model_lock = False
        
        # Auto-select embedding model logic (moved from db.py global)
        self.selected_model = settings.embedding_model_override or "all-MiniLM-L6-v2"
        # Ideally we use select_embedding_model() but for now simplified or imported
        if not settings.embedding_model_override:
            from core.utils.hardware import select_embedding_model
            self.selected_model, _ = select_embedding_model()

    def _create_model(self, path_or_name: str, device: str) -> SentenceTransformer:
        """Helper to instantiate the model."""
        return SentenceTransformer(
            path_or_name,
            device=device,
            trust_remote_code=True,
            model_kwargs={"torch_dtype": torch.float16} 
            if device == "cuda" else {},
        )

    def load(self, force_reload: bool = False) -> SentenceTransformer:
        """Loads the SentenceTransformer model."""
        if self._model and not force_reload:
            self._last_used = time.time()
            return self._model

        # Determine target device
        device = settings.device
        if device == "cuda" and not torch.cuda.is_available():
            log("CUDA not available, falling back to CPU for Encoders")
            device = "cpu"

        cache_dir = settings.cache_dir / "models" / "embeddings"
        cache_dir.mkdir(parents=True, exist_ok=True)
        
        # Determine local path
        model_name_safe = self.selected_model.replace("/", "_")
        local_model_path = cache_dir / model_name_safe

        try:
            # Try loading from local path first
            if local_model_path.exists():
                try:
                    log(f"Loading embedding model from local cache: {local_model_path}")
                    self._model = self._create_model(str(local_model_path), device)
                    self._last_used = time.time()
                    return self._model
                except Exception as e:
                    log(f"Local model corrupt, re-downloading: {e}")
                    shutil.rmtree(local_model_path)
            
            # Download if missing/corrupt
            log(f"Downloading valid model snapshot for {self.selected_model}...")
            snapshot_path = snapshot_download(
                repo_id=self.selected_model,
                local_dir=local_model_path,
                local_dir_use_symlinks=False,
                resume_download=True  # Helpful for bad connections
            )
            self._model = self._create_model(snapshot_path, device)
            log(f"Model loaded successfully on {device}")

        except Exception as e:
            log(f"Failed to load model {self.selected_model}: {e}")
            # Fallback to base model
            try:
                fallback = "all-MiniLM-L6-v2"
                log(f"Attempting fallback to {fallback}...")
                self._model = self._create_model(fallback, "cpu")
            except Exception as e2:
                log(f"Critical: Failed to load fallback model: {e2}")
                raise e

        self._last_used = time.time()
        return self._model

    def to_cpu(self):
        """Moves model to CPU."""
        if self._model and hasattr(self._model, "to"):
            try:
                self._model.to("cpu")
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            except Exception as e:
                log(f"Error moving encoder to CPU: {e}")

    def to_gpu(self):
        """Moves model to GPU."""
        if self._model and hasattr(self._model, "to") and settings.device == "cuda":
             try:
                self._model.to("cuda")
             except Exception as e:
                 log(f"Failed to move encoder to GPU: {e}")
                 self._model.to("cpu")

    def unload_if_idle(self, timeout: int = 300) -> bool:
        """Unsets the model if idle."""
        if not self._model:
            return False
            
        if time.time() - self._last_used > timeout:
            log("Unloading idle encoder model to free RAM")
            self.to_cpu() # Move to CPU at least, or delete?
            # Original code might have kept it on CPU or deleted it. 
            # Let's delete it to be safe for mixed VRAM usage.
            del self._model
            self._model = None
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            return True
        return False
