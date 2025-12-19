"""Configuration settings for the ASR and LLM pipeline."""

import sys
from enum import Enum
from pathlib import Path
from typing import Literal

import torch
from pydantic import Field, SecretStr, computed_field
from pydantic_settings import BaseSettings, SettingsConfigDict


class LLMProvider(str, Enum):
    """Supported LLM providers."""

    GEMINI = "gemini"
    OLLAMA = "ollama"


class Settings(BaseSettings):
    """Application settings, hardware config, and external keys."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    @staticmethod
    def project_root(start: Path | None = None) -> Path:
        """Find the project root directory."""
        start = start or Path(__file__).resolve()
        for parent in start.parents:
            if (parent / ".git").exists() or (parent / "pyproject.toml").exists():
                return parent
        raise RuntimeError("Project root not found")

    @computed_field
    @property
    def cache_dir(self) -> Path:
        """Central location for all caches (__pycache__, models, temp files)."""
        path = self.project_root() / ".cache"
        path.mkdir(exist_ok=True)
        return path

    @computed_field
    @property
    def model_cache_dir(self) -> Path:
        """Directory for model weights."""
        path = self.project_root() / "models"
        path.mkdir(exist_ok=True)
        return path

    @computed_field
    @property
    def prompt_dir(self) -> Path:
        """Directory for prompt templates."""
        path = self.project_root() / "prompts"
        path.mkdir(exist_ok=True)
        return path

    @computed_field
    @property
    def log_dir(self) -> Path:
        """Path to project_root/logs."""
        path = self.project_root() / "logs"
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

    gemini_api_key: SecretStr | None = Field(
        default=None, validation_alias="GOOGLE_API_KEY"
    )
    gemini_model: str = "gemini-1.5-flash"

    ollama_base_url: str = "http://localhost:11434"
    ollama_model: str = "llava:7b"

    tmdb_api_key: str | None = None
    omdb_api_key: str | None = None
    hf_token: str | None = Field(default=None, validation_alias="HF_TOKEN")

    frame_interval: int = Field(default=15, description="Seconds between frames")
    batch_size: int = Field(default=24)
    device_override: Literal["cuda", "cpu", "mps"] | None = None

    language: str | None = "ta"
    whisper_model_map: dict[str, list[str]] = {
        "ta": ["large-v3", "large-v2"],
        "en": ["medium.en", "small.en"],
    }
    fallback_model_id: str = "medium"

    # Voice Intelligence
    enable_voice_analysis: bool = True
    pyannote_model: str = "pyannote/speaker-diarization-3.1"
    voice_embedding_model: str = "pyannote/wespeaker-voxceleb-resnet34-LM"
    min_speakers: int | None = None
    max_speakers: int | None = None

    # Resource
    enable_resource_monitoring: bool = True
    max_cpu_percent: float = 90.0
    max_ram_percent: float = 85.0
    max_temp_celsius: float = 85.0  # Pause if CPU hits 85°C

    # Pause duration when overheated (seconds)
    cool_down_seconds: int = 30

    # Langfuse Configuration
    langfuse_backend: Literal["docker", "cloud", "disabled"] = Field(
        default="disabled",
        description="Langfuse backend selection",
    )

    # Cloud Langfuse
    langfuse_public_key: str | None = None
    langfuse_secret_key: str | None = None
    langfuse_host: str = "https://cloud.langfuse.com"

    # Local (Docker) Langfuse
    langfuse_docker_host: str = "http://localhost:3000"

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

    @computed_field
    @property
    def compute_type(self) -> str:
        """Determine the compute type (float16/int8) based on device."""
        if self.device == "cuda":
            return "float16"
        return "int8"

    @computed_field
    @property
    def device_index(self) -> list[int]:
        """List of available device indices."""
        if self.device == "cuda":
            return list(range(torch.cuda.device_count()))
        return []


settings = Settings()

sys.pycache_prefix = str(settings.cache_dir / "pycache")
