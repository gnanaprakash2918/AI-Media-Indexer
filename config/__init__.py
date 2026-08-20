import os
import sys

from .hardware import HardwareProfile, get_hardware_profile
from .llm import LLMProvider
from .settings import Settings

settings = Settings()

# Centralize ALL model downloads to project's models/ directory
# Must be set BEFORE any HuggingFace/Transformers imports elsewhere
_hf_cache = str(settings.model_cache_dir / "huggingface")
os.environ["HF_HOME"] = _hf_cache
os.environ["HF_HUB_CACHE"] = _hf_cache  # Explicit hub cache location
os.environ["TORCH_HOME"] = str(settings.model_cache_dir / "torch")
os.environ["XDG_CACHE_HOME"] = str(settings.model_cache_dir)

sys.pycache_prefix = str(settings.cache_dir / "pycache")

__all__ = ["Settings", "settings", "HardwareProfile", "LLMProvider", "get_hardware_profile"]
