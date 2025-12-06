"""Ollama LLM adapter.

Wraps the `ollama` async client and adapts it to the LLMInterface.
"""

import os
from pathlib import Path
from typing import Any

import ollama

from .interface import LLMInterface, T


class OllamaLLM(LLMInterface):
    """Adapter for the Ollama LLM service."""

    def __init__(
        self,
        model_name: str | None = None,
        base_url_env: str = "OLLAMA_BASE_URL",
        prompt_dir: str = "./prompts",
    ):
        """Initialize the Ollama AsyncClient.

        Args:
            model_name: Optional model override.
            base_url_env: Environment variable name for the base URL.
            prompt_dir: Prompt template directory.
        """
        super().__init__(prompt_dir=prompt_dir)

        self.model = model_name or os.getenv("OLLAMA_MODEL", "llava:7b")
        base_url = os.getenv(base_url_env, "http://localhost:11434")

        print(
            f"Initializing Ollama client with model={self.model}, base_url={base_url}"
        )

        try:
            self.client = ollama.AsyncClient(host=base_url)
            print("Ollama AsyncClient initialized successfully.")
        except Exception as e:
            print(f"Failed to construct Ollama AsyncClient: {e}")
            raise RuntimeError(f"Failed to construct Ollama AsyncClient: {e}") from e

    async def generate(self, prompt: str, **kwargs: Any) -> str:
        """Generate text from the Ollama service for a user prompt."""
        debug_prompt = prompt[:100] + "..." if len(prompt) > 100 else prompt
        print(f"Ollama generating text for prompt: {debug_prompt}")

        try:
            resp = await self.client.chat(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                options={"temperature": kwargs.get("temperature", 0.0)},
            )
            content = resp.get("message", {}).get("content")
            return str(content) if content else ""
        except Exception as e:
            print(f"Ollama generation failed: {e}")
            raise RuntimeError(f"Ollama generation failed: {e}") from e

    async def generate_structured(
        self,
        schema: type[T],
        prompt: str,
        system_prompt: str = "",
        **kwargs: Any,
    ) -> T:
        """Generate structured output and validate it with the given schema."""
        print("Ollama structured generation requested.")

        full_prompt = (
            f"{system_prompt}\nUser Request:\n{prompt}" if system_prompt else prompt
        )

        try:
            response_text = await self.generate(full_prompt, **kwargs)
            return self.parse_json_response(response_text, schema)
        except Exception as e:
            print(f"Ollama structured generation failed: {e}")
            raise RuntimeError(f"Ollama structured generation failed: {e}") from e

    async def describe_image(
        self,
        prompt: str,
        image_path: str | Path,
        system_prompt: str = "",
        **kwargs: Any,
    ) -> str:
        """Generate a description for an image using the Ollama client."""
        try:
            img_path = str(Path(image_path))

            messages = []
            if system_prompt:
                messages.append({"role": "system", "content": system_prompt})

            messages.append(
                {
                    "role": "user",
                    "content": prompt,
                    "images": [img_path],
                }
            )

            resp = await self.client.chat(
                model=self.model,
                messages=messages,
            )

            content = resp.get("message", {}).get("content")
            return str(content) if content else ""
        except Exception as e:
            print(f"Ollama image description failed: {e}")
            raise RuntimeError(f"Ollama image description failed: {e}") from e
