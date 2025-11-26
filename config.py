"""Configuration and environment-backed settings for the LLM and Audio adapters.

This module exposes a small Settings container and an LLMProvider enum used by
the rest of the project to discover which LLM implementation to instantiate.
"""

from __future__ import annotations

import json
import os
from enum import Enum
from pathlib import Path


class LLMProvider(str, Enum):
    """Enumeration of supported LLM providers."""

    GEMINI = "gemini"
    OLLAMA = "ollama"


def _parse_model_map(env_value: str | None) -> dict[str, str]:
    """Parse a compact environment variable into a language->model dict.

    Supports either JSON string
        or simple comma-separated pairs "ta:vasista22/.,en:large-v3".

    Args:
        env_value: Environment variable string.

    Returns:
        Mapping from language code to model identifier.
    """
    if not env_value:
        # Default mapping: Tamil -> vasista22, fallback handled elsewhere.
        return {"ta": "vasista22/whisper-tamil-large-v2", "en": "large-v3"}

    env_value = env_value.strip()
    # Try JSON first.
    try:
        parsed = json.loads(env_value)
        if isinstance(parsed, dict):
            return {k.lower(): str(v) for k, v in parsed.items()}
    except Exception:
        pass

    # Fallback: parse CSV-like pairs
    mapping: dict[str, str] = {}
    try:
        pairs = [p.strip() for p in env_value.split(",") if p.strip()]
        for pair in pairs:
            if ":" in pair:
                k, v = pair.split(":", 1)
                mapping[k.strip().lower()] = v.strip()
    except Exception:
        pass

    # Ensure defaults exist
    if "ta" not in mapping:
        mapping.setdefault("ta", "vasista22/whisper-tamil-large-v2")
    if "en" not in mapping:
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
        WHISPER_MODEL: Default model id for whisper fallback if no mapping matches.
        WHISPER_DEVICE: Preferred device for whisper ("cuda"/"cpu") None to autodetect.
        WHISPER_COMPUTE_TYPE: Optional compute type override ("float16", "int8", ...).
        WHISPER_MODEL_MAP: Language -> model id mapping parsed from env var.
    """

    PROMPT_DIR: Path = Path(os.getenv("PROMPT_DIR", "./prompts"))
    DEFAULT_PROVIDER: LLMProvider = LLMProvider(os.getenv("LLM_PROVIDER", "gemini"))
    DEFAULT_TIMEOUT: int = int(os.getenv("LLM_TIMEOUT", "120"))

    # Gemini
    GEMINI_API_KEY: str | None = os.getenv("GEMINI_API_KEY") or os.getenv(
        "GOOGLE_API_KEY"
    )
    GEMINI_MODEL: str = os.getenv("GEMINI_MODEL", "gemini-1.5-flash")

    # Ollama
    OLLAMA_BASE_URL: str = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
    OLLAMA_MODEL: str = os.getenv("OLLAMA_MODEL", "llava")

    # Whisper (Audio)
    # The default general-purpose model ID used when not choosing via language map.
    WHISPER_MODEL: str = os.getenv("WHISPER_MODEL", "large-v3")
    WHISPER_DEVICE: str | None = os.getenv("WHISPER_DEVICE", None)
    WHISPER_COMPUTE_TYPE: str | None = os.getenv("WHISPER_COMPUTE_TYPE", None)

    # Optional mapping from language (e.g., "ta") to model id. Parse from env var.
    # WHISPER_MODEL_MAP accepts JSON or CSV-style "ta:vasista22/...,en:large-v3"
    WHISPER_MODEL_MAP: dict[str, str] = _parse_model_map(os.getenv("WHISPER_MODEL_MAP"))

    @property
    def whisper_model_map(self) -> dict[str, str]:
        """Return the whisper language->model mapping."""
        return self.WHISPER_MODEL_MAP


settings = Settings()
