import os
from typing import Literal, cast

from .gemini import GeminiLLM
from .interface import LLMInterface
from .ollama import OllamaLLM


class LLMFactory:
    @staticmethod
    def create_llm(
        provider: Literal["gemini", "ollama"] = "gemini",
        prompt_dir: str = "./prompts",
        **kwargs,
    ) -> LLMInterface:
        """
        Factory method to create an LLM instance based on the provider.
        """
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
        """
        Returns the default LLM based on environment variables.
        """
        provider = os.getenv("LLM_PROVIDER", "gemini")
        provider = provider.lower()

        if provider not in ("gemini", "ollama"):
            provider = "gemini"

        return LLMFactory.create_llm(
            cast(Literal["gemini", "ollama"], provider), prompt_dir
        )
