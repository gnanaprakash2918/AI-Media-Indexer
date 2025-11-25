"""Factory helpers to construct LLM implementations."""

import os
from typing import Literal, cast

from .gemini import GeminiLLM
from .interface import LLMInterface
from .ollama import OllamaLLM


class LLMFactory:
    """Factory class that constructs LLMInterface implementations."""

    @staticmethod
    def create_llm(
        provider: Literal["gemini", "ollama"] = "gemini",
        prompt_dir: str = "./prompts",
        **kwargs,
    ) -> LLMInterface:
        """Create an LLM instance for the given provider."""
        provider = cast(Literal["gemini", "ollama"], provider.lower())
        print(f"LLMFactory: Creating LLM for provider '{provider}'")

        if provider == "gemini":
            return GeminiLLM(prompt_dir=prompt_dir, **kwargs)
        elif provider == "ollama":
            return OllamaLLM(prompt_dir=prompt_dir, **kwargs)
        else:
            raise ValueError(f"Unknown LLM provider: {provider}")

    @staticmethod
    def get_default_llm(prompt_dir: str = "./prompts") -> LLMInterface:
        """Return the default LLM based on the LLM_PROVIDER environment variable."""
        provider = os.getenv("LLM_PROVIDER", "gemini")
        provider = provider.lower()

        if provider not in ("gemini", "ollama"):
            provider = "gemini"

        return LLMFactory.create_llm(
            cast(Literal["gemini", "ollama"], provider), prompt_dir
        )
