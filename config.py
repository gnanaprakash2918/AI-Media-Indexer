"""Configuration settings for the ASR and LLM pipeline."""

from enum import Enum
from pathlib import Path
from typing import Literal

import torch
from pydantic import Field, computed_field
from pydantic_settings import BaseSettings, SettingsConfigDict


class LLMProvider(str, Enum):
    """Supported LLM providers."""

    GEMINI = "gemini"
    OLLAMA = "ollama"


class Settings(BaseSettings):
    """Application settings, hardware config, and external keys."""

    model_config = SettingsConfigDict(
        env_file=".env", env_file_encoding="utf-8", extra="ignore", case_sensitive=False
    )

    #  Paths
    @property
    def project_root(self) -> Path:
        """Finds project root by looking for .git or .env files."""
        current = Path(__file__).resolve()
        for parent in current.parents:
            if (parent / ".git").exists() or (parent / ".env").exists():
                return parent
        return current.parent.parent

    @computed_field
    @property
    def model_cache_dir(self) -> Path:
        """Path to project_root/models (primary cache)."""
        path = self.project_root / "models"
        path.mkdir(exist_ok=True)
        return path

    @computed_field
    @property
    def prompt_dir(self) -> Path:
        """Path to project_root/prompts."""
        path = self.project_root / "prompts"
        path.mkdir(exist_ok=True)
        return path

    # LLM Config
    llm_provider: LLMProvider = Field(default=LLMProvider.OLLAMA)
    llm_timeout: int = Field(default=120)
    gemini_api_key: str | None = Field(default=None, validation_alias="GOOGLE_API_KEY")
    gemini_model: str = "gemini-1.5-flash"
    ollama_base_url: str = "http://localhost:11434"
    ollama_model: str = "llava"

    # ASR Config
    fallback_model_id: str = "openai/whisper-large-v3-turbo"
    whisper_model_map: dict[str, list[str]] = Field(
        default={
            "ta": [
                "openai/whisper-large-v2",
                "vasista22/whisper-tamil-large-v2",
                "openai/whisper-large-v3-turbo",
            ],
            "en": [
                "openai/whisper-large-v3-turbo",
                "distil-whisper/distil-large-v3",
            ],
        }
    )

    language: str = Field(default="ta", description="Target language code")
    batch_size: int = Field(default=24, ge=1)
    chunk_length_s: int = Field(default=30, ge=1)
    hf_token: str | None = None

    # Hardware
    device_override: Literal["cuda", "cpu", "mps"] | None = None
    compute_type_override: Literal["float16", "float32"] | None = None

    @computed_field
    @property
    def device(self) -> str:
        """Determines the torch device string."""
        if self.device_override:
            return self.device_override
        if torch.cuda.is_available():
            return "cuda"
        if torch.backends.mps.is_available():
            return "mps"
        return "cpu"

    @computed_field
    @property
    def device_index(self) -> int:
        """Returns integer device index for Pipeline (0 for GPU, -1 for CPU)."""
        return 0 if self.device == "cuda" else -1

    @computed_field
    @property
    def torch_dtype(self) -> torch.dtype:
        """Determines optimal torch data type."""
        if self.compute_type_override == "float32":
            return torch.float32
        if self.compute_type_override == "float16":
            return torch.float16
        return torch.float16 if self.device == "cuda" else torch.float32


settings = Settings()
