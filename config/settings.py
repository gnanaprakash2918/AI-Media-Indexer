import logging
from pathlib import Path

from pydantic import computed_field, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

from .hardware import HardwareSettings
from .infra import InfraSettings
from .ingestion import IngestionSettings
from .llm import LLMSettings
from .search import SearchSettings
from .security import SecuritySettings


class Settings(
    BaseSettings,
    HardwareSettings,
    LLMSettings,
    IngestionSettings,
    SearchSettings,
    InfraSettings,
    SecuritySettings,
):
    """Application settings composed of multiple modular mixins."""

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

    @model_validator(mode="after")
    def adjust_dimensions(self) -> "Settings":
        """Auto-adjust embedding dimensions based on model name."""
        model = self.embedding_model_override.lower()

        if "nv-embed-v2" in model:
            # NV-Embed-v2 is 4096 dim
            if self.text_embedding_dim != 4096:
                logging.info(
                    "Auto-adjusting text_embedding_dim to 4096 for NV-Embed-v2"
                )
                self.text_embedding_dim = 4096

        elif "bge-m3" in model:
            if self.text_embedding_dim != 1024:
                self.text_embedding_dim = 1024
        elif "mxbai" in model:
            if self.text_embedding_dim != 1024:
                self.text_embedding_dim = 1024

        return self
