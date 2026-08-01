"""Factory helpers to construct LLM implementations.

=============================================================================
PROVIDER SELECTION (Phase 1 — Model Roster Cleanup)
=============================================================================

All providers expose the same LLMInterface ABC. Swap providers by changing
a single env var or config setting — no code changes at call sites.

| Provider | Use case                              | Config                            |
|----------|---------------------------------------|-----------------------------------|
| vllm     | DEFAULT — local vLLM or any           | LLM_PROVIDER=vllm                 |
|          | OpenAI-compatible hosted endpoint     | VLLM_BASE_URL=http://localhost:8000|
|          | (Together, Fireworks, Anyscale, etc.) | VLM_ENDPOINT_MODEL_NAME=Qwen/...  |
| gemini   | Cloud Gemini (verification LLM)       | LLM_PROVIDER=gemini               |
|          | Put behind a config flag; not default | GOOGLE_API_KEY=...                |
| ollama   | Local dev convenience (slower).       | LLM_PROVIDER=ollama               |
|          | Use when you have no vLLM instance.   | OLLAMA_BASE_URL=...               |

IMPORTANT: There is NO silent fallback between providers. If vLLM is
unreachable you will get a clear connection error telling you to check
VLLM_BASE_URL or set LLM_PROVIDER=ollama. This is intentional — silent
downgrades mask configuration problems.

For local dev without a GPU/vLLM:
    export LLM_PROVIDER=ollama
    # or set in .env: LLM_PROVIDER=ollama

For production with vLLM:
    export LLM_PROVIDER=vllm
    export VLLM_BASE_URL=http://<your-vllm-host>:8000
    export VLM_ENDPOINT_MODEL_NAME=Qwen/Qwen3-VL-2B-Instruct

=============================================================================
"""

import os
from typing import TYPE_CHECKING, Literal, cast

from .interface import LLMInterface
from .ollama import OllamaLLM

if TYPE_CHECKING:
    pass

# Supported providers
SUPPORTED_PROVIDERS = ("vllm", "gemini", "ollama")

# Default provider — vLLM per AGENTS.md "Serving & orchestration".
# Local dev without vLLM: set LLM_PROVIDER=ollama in .env or environment.
DEFAULT_PROVIDER = "vllm"


class LLMFactory:
    """Factory class that constructs LLMInterface implementations.

    Usage:
        # Use vLLM (default)
        llm = LLMFactory.create_llm()

        # Specify provider explicitly
        llm = LLMFactory.create_llm(provider="gemini")

        # Via environment variable (recommended)
        # export LLM_PROVIDER=vllm   (production)
        # export LLM_PROVIDER=ollama (dev without GPU)
        llm = LLMFactory.get_default_llm()

    Provider selection priority:
        1. provider= argument to create_llm()
        2. LLM_PROVIDER environment variable
        3. DEFAULT_PROVIDER constant (vllm)

    No silent fallback: if vLLM is unreachable you get a clear error.
    """

    @staticmethod
    def create_llm(
        provider: Literal["vllm", "gemini", "ollama"] = "vllm",
        prompt_dir: str = "./prompts",
        **kwargs,
    ) -> LLMInterface:
        """Create an LLM instance for the given provider.

        Args:
            provider: Which LLM provider to use:
                - "vllm"   : vLLM or any OpenAI-compatible endpoint (DEFAULT)
                - "gemini" : Google Gemini API (cloud, requires GOOGLE_API_KEY)
                - "ollama" : Local Ollama (dev convenience, no GPU needed)
            prompt_dir: Directory containing prompt templates.
            **kwargs: Provider-specific options forwarded to the implementation.

        Returns:
            LLMInterface implementation.

        Raises:
            ValueError: For unknown provider names.
            RuntimeError: If vLLM endpoint is unreachable (on first inference).
        """
        provider = cast(
            Literal["vllm", "gemini", "ollama"], provider.lower()
        )
        print(f"[LLMFactory] Creating LLM for provider='{provider}'")

        if provider == "vllm":
            from .vllm import VLLMProvider

            return VLLMProvider(prompt_dir=prompt_dir, **kwargs)

        elif provider == "gemini":
            from .gemini import GeminiLLM

            return GeminiLLM(prompt_dir=prompt_dir, **kwargs)

        elif provider == "ollama":
            return OllamaLLM(prompt_dir=prompt_dir, **kwargs)

        else:
            raise ValueError(
                f"Unknown LLM provider: '{provider}'. "
                f"Supported: {SUPPORTED_PROVIDERS}. "
                f"Set LLM_PROVIDER env var to one of: {SUPPORTED_PROVIDERS}"
            )

    @staticmethod
    def get_default_llm(prompt_dir: str = "./prompts") -> LLMInterface:
        """Return the default LLM from the LLM_PROVIDER environment variable.

        Environment Variables:
            LLM_PROVIDER: "vllm" (default) | "gemini" | "ollama"

        NOTE: There is NO silent fallback. An unknown or unreachable provider
        will raise an error with a clear, actionable message.

        For local dev without a GPU/vLLM instance:
            Linux:   export LLM_PROVIDER=ollama
            Windows: set LLM_PROVIDER=ollama
        """
        provider = os.getenv("LLM_PROVIDER", DEFAULT_PROVIDER).lower()

        if provider not in SUPPORTED_PROVIDERS:
            raise ValueError(
                f"[LLMFactory] Unknown LLM_PROVIDER='{provider}'. "
                f"Supported values: {SUPPORTED_PROVIDERS}. "
                f"Check your .env file or environment variables."
            )

        return LLMFactory.create_llm(
            cast(Literal["vllm", "gemini", "ollama"], provider), prompt_dir
        )

    @staticmethod
    def create_vision_llm(prompt_dir: str = "./prompts") -> LLMInterface:
        """Create the LLM for vision/frame analysis tasks.

        Defaults to vLLM (Qwen3-VL). For dev: set LLM_PROVIDER=ollama.
        """
        return LLMFactory.get_default_llm(prompt_dir)

    @staticmethod
    def create_text_llm(prompt_dir: str = "./prompts") -> LLMInterface:
        """Create the LLM for text tasks (query expansion, planning, etc).

        Defaults to vLLM. For dev: set LLM_PROVIDER=ollama.
        """
        return LLMFactory.get_default_llm(prompt_dir)
