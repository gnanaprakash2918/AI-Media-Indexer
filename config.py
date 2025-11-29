"""Configuration settings for the ASR and LLM pipeline.

Defines strict schema for configuration using Pydantic.
Handles environment variables, hardware detection, and project-root path management.
"""

from enum import Enum
from pathlib import Path
from typing import Dict, List, Literal, Optional

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

    @property
    def project_root(self) -> Path:
        """Finds the project root by looking for .git, .env or common folders."""
        current = Path(__file__).resolve()
        for parent in current.parents:
            if (parent / ".git").exists() or (parent / ".env").exists():
                return parent
        return current.parent.parent

    @computed_field  # type: ignore
    @property
    def model_cache_dir(self) -> Path:
        """Returns path to project_root/models."""
        path = self.project_root / "models"
        path.mkdir(exist_ok=True)
        return path

    @computed_field  # type: ignore
    @property
    def prompt_dir(self) -> Path:
        """Returns path to project_root/prompts."""
        path = self.project_root / "prompts"
        path.mkdir(exist_ok=True)
        return path

    llm_provider: LLMProvider = Field(default=LLMProvider.OLLAMA)
    llm_timeout: int = Field(default=120)

    gemini_api_key: Optional[str] = Field(
        default=None, validation_alias="GOOGLE_API_KEY"
    )
    gemini_model: str = Field(default="gemini-1.5-flash")

    ollama_base_url: str = Field(default="http://localhost:11434")
    ollama_model: str = Field(default="llava")

    fallback_model_id: str = Field(default="large-v3")

    whisper_model_map: Dict[str, List[str]] = Field(
        default={
            "ta": [
                "large-v3",
                "vasista22/whisper-tamil-large-v2",
                "base",
                # "openai/whisper-large-v3",
            ],
            "en": [
                "large-v3",
                # "openai/whisper-large-v3",
                # "distil-whisper/distil-large-v3",
                "small",
            ],
        }
    )

    language: str = Field(default="ta", description="Default target language code")
    batch_size: int = Field(default=8, ge=1)
    chunk_length_s: int = Field(default=30, ge=1)
    hf_token: Optional[str] = Field(default=None)

    device_override: Optional[Literal["cuda", "cpu", "mps"]] = None
    compute_type_override: Optional[Literal["float16", "float32"]] = None

    @computed_field  # type: ignore
    @property
    def device(self) -> str:
        """Determines the device string (cuda/cpu/mps)."""
        if self.device_override:
            return self.device_override
        if torch.cuda.is_available():
            return "cuda"
        if torch.backends.mps.is_available():
            return "mps"
        return "cpu"

    @computed_field  # type: ignore
    @property
    def device_index(self) -> int:
        """Returns integer device index for Pipeline (0 for GPU, -1 for CPU)."""
        return 0 if self.device == "cuda" else -1

    @computed_field  # type: ignore
    @property
    def torch_dtype(self) -> torch.dtype:
        """Determines optimal torch data type."""
        if self.compute_type_override == "float32":
            return torch.float32
        if self.compute_type_override == "float16":
            return torch.float16
        return torch.float16 if self.device == "cuda" else torch.float32


settings = Settings()
