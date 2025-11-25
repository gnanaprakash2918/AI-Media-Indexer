import os
from enum import Enum
from pathlib import Path


class LLMProvider(str, Enum):
    GEMINI = "gemini"
    OLLAMA = "ollama"


class Settings:
    PROMPT_DIR: Path = Path(os.getenv("PROMPT_DIR", "./prompts"))

    DEFAULT_PROVIDER: LLMProvider = LLMProvider(
        os.getenv("LLM_PROVIDER", "gemini")
    )
    DEFAULT_TIMEOUT: int = int(os.getenv("LLM_TIMEOUT", "120"))

    GEMINI_API_KEY: str | None = os.getenv("GEMINI_API_KEY") or os.getenv(
        "GOOGLE_API_KEY"
    )
    GEMINI_MODEL: str = os.getenv("GEMINI_MODEL", "gemini-2.5-pro")

    OLLAMA_BASE_URL: str = os.getenv(
        "OLLAMA_BASE_URL", "http://localhost:11434"
    )
    OLLAMA_MODEL: str = os.getenv("OLLAMA_MODEL", "llava")


settings = Settings()
