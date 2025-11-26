"""Configuration and environment-backed settings for the LLM and Audio adapters.

This module exposes a small Settings container and an LLMProvider enum used by
the rest of the project to discover which LLM implementation to instantiate.
"""

from __future__ import annotations

import json
import os
from enum import Enum
from pathlib import Path

# Enable hf_transfer for max speed
os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "1"


class LLMProvider(str, Enum):
    """Enumeration of supported LLM providers."""

    GEMINI = "gemini"
    OLLAMA = "ollama"


def _parse_model_map(env_value: str | None) -> dict[str, str]:
    """Parse a compact environment variable into a mapping of language -> model.

    Supports:
        * JSON: {"ta": "vasista22/...", "en": "large-v3"}
        * CSV: "ta:vasista22/whisper-tamil-large-v2,en:large-v3"

    Args:
        env_value: Raw value of WHISPER_MODEL_MAP from environment.

    Returns:
        A dictionary mapping language codes to model identifiers.
    """
    if not env_value:
        # Default mapping.
        return {
            "ta": "vasista22/whisper-tamil-large-v2",
            "en": "large-v3",
        }

    value = env_value.strip()

    # Attempt JSON first.
    try:
        parsed_json = json.loads(value)
        if isinstance(parsed_json, dict):
            return {k.lower(): str(v) for k, v in parsed_json.items()}
    except Exception:
        pass

    # Fallback: CSV-like parsing.
    mapping: dict[str, str] = {}
    try:
        items = [part.strip() for part in value.split(",") if part.strip()]
        for item in items:
            if ":" in item:
                lang, model = item.split(":", 1)
                mapping[lang.strip().lower()] = model.strip()
    except Exception:
        pass

    # Ensure Tamil + English defaults exist.
    mapping.setdefault("ta", "vasista22/whisper-tamil-large-v2")
    mapping.setdefault("en", "large-v3")
    return mapping


class Settings:
    """Runtime configuration values loaded from environment variables.

    Attributes:
        PROMPT_DIR: Directory where prompt templates are stored.
        DEFAULT_PROVIDER: Default LLM provider to use.
        DEFAULT_TIMEOUT: Default request timeout in seconds.
        GEMINI_API_KEY: API key for Gemini (or GOOGLE_API_KEY fallback).
        GEMINI_MODEL: Default Gemini model name.
        OLLAMA_BASE_URL: Base URL for Ollama service.
        OLLAMA_MODEL: Default Ollama model name.
        WHISPER_MODEL: Default whisper model for general-purpose transcription.
        WHISPER_DEVICE: Preferred device override ("cuda" or "cpu").
        WHISPER_COMPUTE_TYPE: Compute precision override for Whisper.
        WHISPER_MODEL_MAP: Mapping of language code -> Whisper model id.
    """

    PROMPT_DIR: Path = Path(os.getenv("PROMPT_DIR", "./prompts"))
    DEFAULT_PROVIDER: LLMProvider = LLMProvider(
        os.getenv("LLM_PROVIDER", "gemini").lower()
    )
    DEFAULT_TIMEOUT: int = int(os.getenv("LLM_TIMEOUT", "120"))

    # Gemini
    GEMINI_API_KEY: str | None = os.getenv("GEMINI_API_KEY") or os.getenv(
        "GOOGLE_API_KEY"
    )
    GEMINI_MODEL: str = os.getenv("GEMINI_MODEL", "gemini-1.5-flash")

    # Ollama
    OLLAMA_BASE_URL: str = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
    OLLAMA_MODEL: str = os.getenv("OLLAMA_MODEL", "llava")

    # Whisper defaults
    WHISPER_MODEL: str = os.getenv("WHISPER_MODEL", "large-v3")
    WHISPER_DEVICE: str | None = os.getenv("WHISPER_DEVICE")
    WHISPER_COMPUTE_TYPE: str | None = os.getenv("WHISPER_COMPUTE_TYPE")

    # Language-based mapping (Tamil & English by default)
    WHISPER_MODEL_MAP: dict[str, str] = _parse_model_map(os.getenv("WHISPER_MODEL_MAP"))

    @property
    def whisper_model_map(self) -> dict[str, str]:
        """Return whisper language → model mapping."""
        return self.WHISPER_MODEL_MAP


settings = Settings()
