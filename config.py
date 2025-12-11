"""Configuration settings for the ASR and LLM pipeline."""

import sys
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

    # Paths
    @staticmethod
    def project_root() -> Path:
        """Finds project root by looking for .git or .env files."""
        current = Path(__file__).resolve()
        for parent in current.parents:
            if (parent / ".git").exists() or (parent / "pyproject.toml").exists():
                return parent
        return current.parent.parent

    @computed_field
    @property
    def cache_dir(self) -> Path:
        """Central location for all caches (__pycache__, models, temp files)."""
        path = Settings.project_root() / ".cache"
        path.mkdir(exist_ok=True)
        return path

    @computed_field
    @property
    def model_cache_dir(self) -> Path:
        """Path to project_root/models (primary cache)."""
        path = self.cache_dir / "models"
        path.mkdir(exist_ok=True)
        return path

    @computed_field
    @property
    def log_dir(self) -> Path:
        """Path to project_root/logs."""
        path = Settings.project_root() / "logs"
        path.mkdir(exist_ok=True)
        return path

    #  Infrastructure (Qdrant)
    qdrant_host: str = Field(default="localhost", description="Qdrant host")
    qdrant_port: int = Field(default=6333, description="Qdrant HTTP port")
    qdrant_backend: str = Field(default="docker", description="'memory' or 'docker'")

    #  Agent & LLM
    agent_model: str = Field(default="llama3.1", description="Model for Agent CLI")
    llm_provider: LLMProvider = Field(default=LLMProvider.OLLAMA)
    ollama_base_url: str = Field(default="http://localhost:11434")
    ollama_vision_model: str = Field(default="llava:7b")

    #  Keys & External APIs
    tmdb_api_key: str | None = None
    omdb_api_key: str | None = None
    hf_token: str | None = None

    #  Ingestion Settings
    frame_interval: int = Field(default=15, description="Seconds between frames")
    batch_size: int = Field(default=24)
    device_override: Literal["cuda", "cpu", "mps"] | None = None

    #  Whisper Configuration
    whisper_model_map: dict[str, list[str]] = Field(
        default={
            "ta": ["openai/whisper-large-v3-turbo", "openai/whisper-large-v2"],
            "en": ["openai/whisper-large-v3-turbo", "distil-whisper/distil-large-v3"],
        }
    )
    fallback_model_id: str = "openai/whisper-large-v3-turbo"
    language: str = Field(default="en", description="Target language code")

    @computed_field
    @property
    def prompt_dir(self) -> Path:
        """Path to project_root/prompts."""
        path = Settings.project_root() / "prompts"
        return path

    @computed_field
    @property
    def device(self) -> str:
        """Decide the device based on CPU or CUDA."""
        if self.device_override:
            return self.device_override
        if torch.cuda.is_available():
            return "cuda"
        if torch.backends.mps.is_available():
            return "mps"
        return "cpu"


settings = Settings()

#  Auto-Cleanup PyCache
sys.pycache_prefix = str(settings.cache_dir / "pycache")
