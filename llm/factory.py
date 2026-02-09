"""Factory helpers to construct LLM implementations.

=============================================================================
LLM PROVIDER COMPARISON & RECOMMENDATIONS
=============================================================================

| Task                    | Best Provider     | Why                           |
|-------------------------|-------------------|-------------------------------|
| VISION/FRAME ANALYSIS   | Gemini 1.5 Pro    | Best visual understanding     |
|                         | GPT-4o            | Close second, faster          |
|                         | Ollama (llava)    | FREE, good for dev/testing    |
|-------------------------|-------------------|-------------------------------|
| STRUCTURED OUTPUT       | GPT-4o            | Native JSON schema support    |
|                         | Gemini 1.5        | Good with prompting           |
|                         | Ollama (mistral)  | FREE, decent with prompting   |
|-------------------------|-------------------|-------------------------------|
| QUERY EXPANSION         | Gemini 1.5 Pro    | Best entity extraction        |
|                         | Claude 3.5        | Excellent reasoning           |
|                         | Ollama (llama3)   | FREE, good for simple queries |
|-------------------------|-------------------|-------------------------------|
| RE-RANKING/REASONING    | Claude 3.5 Opus   | Best chain-of-thought         |
|                         | GPT-4o            | Fast and accurate             |
|                         | Ollama (llama3.1) | FREE, acceptable quality      |
|-------------------------|-------------------|-------------------------------|
| SPEED (LOW LATENCY)     | Gemini 1.5 Flash  | ~100ms response               |
|                         | GPT-4o-mini       | Very fast, cheap              |
|                         | Ollama (local)    | No network latency            |
|-------------------------|-------------------|-------------------------------|
| COST                    | Ollama (local)    | FREE (GPU only)               |
|                         | Gemini Flash      | Very cheap                    |
|                         | GPT-4o-mini       | Budget cloud option           |

RECOMMENDATION FOR PRODUCTION ($289B QUALITY):
- Vision: Gemini 1.5 Pro or GPT-4o (best accuracy)
- Structured: GPT-4o with json_schema response_format
- Re-ranking: Claude 3.5 Opus for chain-of-thought reasoning

RECOMMENDATION FOR DEVELOPMENT/TESTING:
- All tasks: Ollama with llava/llama3 (free, local, fast iteration)

=============================================================================
"""

import os
from typing import TYPE_CHECKING, Literal, cast

from .interface import LLMInterface
from .ollama import OllamaLLM

# Lazy import for Gemini to avoid triggering google-generativeai SDK errors
# when user is not using Gemini
if TYPE_CHECKING:
    from .gemini import GeminiLLM

# Supported providers - easily add new ones here
SUPPORTED_PROVIDERS = ("gemini", "ollama", "openai", "anthropic")

# Default provider - change this to swap globally
DEFAULT_PROVIDER = "ollama"


class LLMFactory:
    """Factory class that constructs LLMInterface implementations.

    Usage:
        # Use default (Ollama)
        llm = LLMFactory.create_llm()

        # Specify provider
        llm = LLMFactory.create_llm(provider="gemini")

        # Use environment variable
        # export LLM_PROVIDER=gemini
        llm = LLMFactory.get_default_llm()

    To swap providers:
        1. Set LLM_PROVIDER environment variable, OR
        2. Change DEFAULT_PROVIDER constant above, OR
        3. Pass provider= argument to create_llm()
    """

    @staticmethod
    def create_llm(
        provider: Literal["gemini", "ollama"] = "ollama",
        prompt_dir: str = "./prompts",
        **kwargs,
    ) -> LLMInterface:
        """Create an LLM instance for the given provider.

        Args:
            provider: Which LLM to use. Options:
                - "ollama": Local Ollama (FREE, good for dev)
                - "gemini": Google Gemini API (best vision)
                - Future: "openai", "anthropic"
            prompt_dir: Directory containing prompt templates.
            **kwargs: Provider-specific options (model name, etc.)

        Returns:
            LLMInterface implementation.

        Best Practices:
            - Use Ollama for development/testing (free, fast iteration)
            - Use Gemini/GPT-4o for production vision tasks (best accuracy)
            - Use Claude for complex reasoning/re-ranking (best CoT)
        """
        provider = cast(Literal["gemini", "ollama"], provider.lower())
        print(f"LLMFactory: Creating LLM for provider '{provider}'")

        if provider == "gemini":
            # BEST FOR: Vision, frame analysis, structured extraction
            # COST: ~$0.00025/image (cheap)
            # SPEED: Fast (~1-2s for vision)
            # Lazy import to avoid triggering google-generativeai SDK errors
            from .gemini import GeminiLLM
            return GeminiLLM(prompt_dir=prompt_dir, **kwargs)

        elif provider == "ollama":
            # BEST FOR: Development, testing, cost-sensitive production
            # COST: FREE (local GPU only)
            # SPEED: Depends on hardware (~2-5s for vision)
            # MODELS: llava (vision), llama3 (text), mistral (structured)
            return OllamaLLM(prompt_dir=prompt_dir, **kwargs)

        # TODO: Add more providers for SOTA quality
        # elif provider == "openai":
        #     # BEST FOR: Structured output (json_schema), fast vision
        #     # COST: ~$0.01/image (expensive)
        #     return OpenAILLM(prompt_dir=prompt_dir, **kwargs)
        #
        # elif provider == "anthropic":
        #     # BEST FOR: Complex reasoning, chain-of-thought, re-ranking
        #     # COST: ~$0.015/1K tokens (expensive)
        #     return AnthropicLLM(prompt_dir=prompt_dir, **kwargs)

        else:
            raise ValueError(
                f"Unknown LLM provider: {provider}. Supported: {SUPPORTED_PROVIDERS}"
            )

    @staticmethod
    def get_default_llm(prompt_dir: str = "./prompts") -> LLMInterface:
        """Return the default LLM based on the LLM_PROVIDER environment variable.

        Environment Variables:
            LLM_PROVIDER: Provider name (ollama, gemini, openai, anthropic)
                         Default: "ollama" (free, local)

        To swap providers easily:
            Windows: set LLM_PROVIDER=gemini
            Linux:   export LLM_PROVIDER=gemini
        """
        provider = os.getenv("LLM_PROVIDER", DEFAULT_PROVIDER)
        provider = provider.lower()

        if provider not in ("gemini", "ollama"):
            print(
                f"Warning: Unknown provider '{provider}', falling back to ollama"
            )
            provider = "ollama"

        return LLMFactory.create_llm(
            cast(Literal["gemini", "ollama"], provider), prompt_dir
        )

    @staticmethod
    def create_vision_llm(prompt_dir: str = "./prompts") -> LLMInterface:
        """Create the best LLM for vision/frame analysis tasks.

        For SOTA accuracy, use Gemini 1.5 Pro or GPT-4o.
        For free development, use Ollama with llava.
        """
        # Check for production flag
        if os.getenv("PRODUCTION_MODE", "false").lower() == "true":
            # Production: Use best available
            if os.getenv("GEMINI_API_KEY"):
                return LLMFactory.create_llm("gemini", prompt_dir)

        # Default: Ollama (free)
        return LLMFactory.create_llm("ollama", prompt_dir)

    @staticmethod
    def create_text_llm(prompt_dir: str = "./prompts") -> LLMInterface:
        """Create the best LLM for text tasks (query expansion, etc).

        For SOTA accuracy, use Claude 3.5 or GPT-4o.
        For free development, use Ollama with llama3.
        """
        return LLMFactory.get_default_llm(prompt_dir)
