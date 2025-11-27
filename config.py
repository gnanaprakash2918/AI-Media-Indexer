"""Configuration and environment-backed settings for the LLM and Audio adapters.

This module exposes a small Settings container and an LLMProvider enum.
"""

from __future__ import annotations

import json
import os
from enum import Enum
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

# Enable hf_transfer for max speed
os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "1"


class LLMProvider(str, Enum):
    """Enumeration of supported LLM providers."""

    GEMINI = "gemini"
    OLLAMA = "ollama"


def _parse_model_map(env_value: str | None) -> dict[str, list[str]]:
    """Parse env vars into a mapping of language -> list of candidate models.

    Hierarchy for Tamil ('ta'):
      1. TIER 0 (True SOTA): "ai4bharat/indicconformer_stt_ta_hybrid_ctc_rnnt_large"
         - Engine: NVIDIA NeMo
         - Note: Complex install. Best accuracy for native Indian accents.
      2. TIER 1 (Practical SOTA): "jiviai/audioX-south-v1"
         - Engine: Transformers
         - Note: Gated (Requires HF_TOKEN). Fine-tuned Whisper V3.
      3. TIER 2 (Robust Open): "vasista22/whisper-tamil-large-v2"
         - Engine: Transformers / Faster-Whisper
         - Note: Standard 'IndicWhisper'.
      4. TIER 3 (Fallback): "large-v3"
         - Engine: Faster-Whisper

    Args:
        env_value: Raw value from environment.

    Returns:
        Dictionary of language code to list of model IDs.
    """
    mapping: dict[str, list[str]] = {
        "en": ["large-v3"],
        "ta": [
            # The True SOTA (NeMo)
            # "ai4bharat/indicconformer_stt_ta_hybrid_ctc_rnnt_large",
            "ai4bharat/indic-conformer-600m-multilingual",
            # The Practical SOTA (Transformers)
            "jiviai/audioX-south-v1",
            # The Reliable Open (Transformers/FW)
            "vasista22/whisper-tamil-large-v2",
            # Fallback
            "openai/whisper-large-v3",
        ],
    }

    if not env_value:
        return mapping

    value = env_value.strip()

    try:
        parsed_json = json.loads(value)
        if isinstance(parsed_json, dict):
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

    # HF Token (Required for Tier 1 SOTA models like JiviAI)
    HF_TOKEN: str | None = os.getenv("HF_TOKEN") or os.getenv("HUGGINGFACE_HUB_TOKEN")

    # Model Mapping
    WHISPER_MODEL_MAP: dict[str, list[str]] = _parse_model_map(
        os.getenv("WHISPER_MODEL_MAP")
    )

    @property
    def whisper_model_map(self) -> dict[str, list[str]]:
        """Return whisper language → [model_candidates] mapping."""
        return self.WHISPER_MODEL_MAP


settings = Settings()
