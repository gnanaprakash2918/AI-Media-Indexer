"""Configuration settings for the ASR and LLM pipeline.

This module defines the strict schema for configuration, handling environment
variables, default values, and hardware detection automatically.
"""

from enum import Enum
from pathlib import Path
from typing import Dict, List, Literal

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

    # --- Paths ---
    prompt_dir: Path = Field(default=Path("./prompts"))

    # --- LLM Configuration ---
    llm_provider: LLMProvider = Field(default=LLMProvider.GEMINI)
    llm_timeout: int = Field(default=120)

    # Gemini
    gemini_api_key: str | None = Field(default=None, validation_alias="GOOGLE_API_KEY")
    gemini_model: str = Field(default="gemini-1.5-flash")

    # Ollama
    ollama_base_url: str = Field(default="http://localhost:11434")
    ollama_model: str = Field(default="llava")

    # --- ASR / Whisper Configuration ---
    # Default fallback model if map lookup fails
    model_id: str = Field(default="vasista22/whisper-tamil-large-v2")

    # Tiered Model Map (JSON or Dict structure supported via env vars)
    whisper_model_map: Dict[str, List[str]] = Field(
        default={
            "ta": [
                "vasista22/whisper-tamil-large-v2",  # Best Open
                "jiviai/audioX-south-v1",  # SOTA (Gated)
                "large-v3",  # Fallback
            ],
            "en": ["openai/whisper-large-v3", "distil-whisper/distil-large-v3"],
        }
    )

    # Core ASR Params
    language: str = Field(default="ta", description="Target language code")
    batch_size: int = Field(default=8, ge=1)
    chunk_length_s: int = Field(default=30, ge=1)
    hf_token: str | None = Field(default=None)

    # --- Hardware Overrides ---
    device_override: Literal["cuda", "cpu", "mps", None] = None
    compute_type_override: Literal["float16", "float32", None] = None

    @computed_field  # type: ignore
    @property
    def device(self) -> str:
        """Determines the best available hardware device."""
        if self.device_override:
            return self.device_override
        if torch.cuda.is_available():
            return "cuda"
        if torch.backends.mps.is_available():
            return "mps"
        return "cpu"

    @computed_field  # type: ignore
    @property
    def torch_dtype(self) -> torch.dtype:
        """Determines the optimal torch data type."""
        if self.compute_type_override == "float32":
            return torch.float32
        if self.compute_type_override == "float16":
            return torch.float16
        return torch.float16 if self.device == "cuda" else torch.float32


# Global settings instance
settings = Settings()
