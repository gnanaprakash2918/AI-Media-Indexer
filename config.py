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


def _parse_model_map(env_value: str | None) -> dict[str, list[str]]:
    """Parse env variable into a mapping of language -> list of candidate models.

    The values are lists to support fallback hierarchies (Best -> Worst).

    Hierarchy Strategy for Tamil ('ta') - Performance Based:
      1. Tier 1 (Best WER): "ai4bharat/indicwav2vec-tamil"
         - Architecture: Wav2Vec2 (Transformers)
         - Notes: SOTA accuracy for Indian languages.
      2. Tier 2 (Best Whisper): "jiviai/audioX-south-v1"
         - Architecture: Whisper (Faster-Whisper)
         - Notes: GATED model. High accuracy, includes punctuation.
      3. Tier 3 (Reliable Open): "vasista22/whisper-tamil-large-v2"
         - Architecture: Whisper (Faster-Whisper)
         - Notes: Reliable, ungated, good instruction following.
      4. Tier 4 (Fallback): "openai/whisper-large-v3"
         - Architecture: Whisper
         - Notes: Generic fallback.

    Args:
        env_value: Raw value of WHISPER_MODEL_MAP from environment.

    Returns:
        A dictionary mapping language codes to a LIST of model identifiers.
    """
    mapping: dict[str, list[str]] = {
        "en": ["large-v3"],
        "ta": [
            "ai4bharat/indicwav2vec-tamil",
            "jiviai/audioX-south-v1",
            "vasista22/whisper-tamil-large-v2",
            "large-v3",
        ],
    }

    if not env_value:
        return mapping

    value = env_value.strip()

    # Attempt JSON first.
    try:
        parsed_json = json.loads(value)
        if isinstance(parsed_json, dict):
            # Convert single strings to lists if needed
            cleaned: dict[str, list[str]] = {}
            for k, v in parsed_json.items():
                if isinstance(v, str):
                    cleaned[k.lower()] = [v]
                elif isinstance(v, list):
                    cleaned[k.lower()] = [str(x) for x in v]
            mapping.update(cleaned)
            return mapping
    except Exception:
        pass

    # Fallback: CSV-like parsing "ta:model1,en:model2" (Only supports 1 model per lang)
    try:
        items = [part.strip() for part in value.split(",") if part.strip()]
        for item in items:
            if ":" in item:
                lang, model = item.split(":", 1)
                mapping[lang.strip().lower()] = [model.strip()]
    except Exception:
        pass

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
        HF_TOKEN: Hugging Face token for gated models.
        WHISPER_MODEL_MAP: Mapping of language code -> List of models.
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

    # Hugging Face Token (Needed for JiviAI models)
    HF_TOKEN: str | None = os.getenv("HF_TOKEN") or os.getenv("HUGGINGFACE_HUB_TOKEN")

    # Language-based mapping
    WHISPER_MODEL_MAP: dict[str, list[str]] = _parse_model_map(
        os.getenv("WHISPER_MODEL_MAP")
    )

    @property
    def whisper_model_map(self) -> dict[str, list[str]]:
        """Return whisper language → [model_candidates] mapping."""
        return self.WHISPER_MODEL_MAP


settings = Settings()
