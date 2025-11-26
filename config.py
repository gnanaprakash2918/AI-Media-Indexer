"""Configuration and environment-backed settings for the LLM and Audio adapters.

This module exposes a small Settings container and an LLMProvider enum used by
the rest of the project to discover which LLM implementation to instantiate.
"""

import os
from enum import Enum
from pathlib import Path


class LLMProvider(str, Enum):
    """Enumeration of supported LLM providers."""

    GEMINI = "gemini"
    OLLAMA = "ollama"


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
    # To use 'vasista22/whisper-tamil-large-v2', you must convert it first
    WHISPER_MODEL: str = os.getenv("WHISPER_MODEL", "large-v3")
    WHISPER_DEVICE: str | None = os.getenv("WHISPER_DEVICE", None)  # Auto-detect
    WHISPER_COMPUTE_TYPE: str | None = os.getenv("WHISPER_COMPUTE_TYPE", None)


settings = Settings()
