"""Gemini LLM integration using langchain-google-genai.

This module adapts the ChatGoogleGenerativeAI client into the project's
LLMInterface.
"""

import asyncio
import base64
import mimetypes
import os
from pathlib import Path
from typing import Any, cast

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_google_genai import ChatGoogleGenerativeAI

from .interface import LLMInterface, T


class GeminiLLM(LLMInterface):
    """Adapter for the Google Gemini (Generative AI) LLM."""

    def __init__(
        self,
        model_name: str | None = None,
        api_key_env: str = "GEMINI_API_KEY",
        prompt_dir: str = "prompts",
        timeout: int = 60,
    ):
        """Initialize the Gemini client.

        Args:
            model_name: Optional model name to override env.
            api_key_env: Name of the environment variable having the API key.
            prompt_dir: Directory for prompt templates.
            timeout: Request timeout in seconds.
        """
        print("Initializing Gemini LLM Interface")
        super().__init__(prompt_dir=prompt_dir)

        model = model_name or os.getenv("GEMINI_MODEL", "gemini-2.5-pro")
        api_key = os.getenv(api_key_env) or os.getenv("GOOGLE_API_KEY")

        print(f"Using Gemini model: {model}")
        if not api_key:
            print("Gemini API key not set.")
            raise ValueError("Gemini API key not set.")

        print("Gemini API key set successfully.")
        self.llm = ChatGoogleGenerativeAI(
            model=model,
            api_key=api_key,
            request_timeout=timeout,
            temperature=0.0,
        )

    async def generate(self, prompt: str, **kwargs: Any) -> str:
        """Generate text for a plain prompt and return it as a string."""
        try:
            debug_prompt = prompt[:100] + "..." if len(prompt) > 100 else prompt
            print(f"Generating text with prompt: {debug_prompt}")

            response = await self.llm.ainvoke(prompt)
            content = getattr(response, "content", str(response))

            print("Generated content using Gemini")
            return str(content)

        except Exception as e:
            raise RuntimeError(f"Gemini generation failed: {e}") from e

    async def generate_structured(
        self,
        schema: type[T],
        prompt: str,
        system_prompt: str = "",
        **kwargs: Any,
    ) -> T:
        """Generate structured output validated against a Pydantic schema.

        If the structured output feature is not available or validation fails,
        a fallback to manual JSON parsing is attempted.
        """
        try:
            debug_prompt = prompt[:100] + "..." if len(prompt) > 100 else prompt
            print(f"Generating structured output with prompt: {debug_prompt}")

            structured_llm = self.llm.with_structured_output(schema)

            messages = []
            if system_prompt:
                messages.append(SystemMessage(content=system_prompt))

            messages.append(HumanMessage(content=prompt))

            result = await structured_llm.ainvoke(messages)
            return cast(T, result)

        except Exception as e:
            print(f"Structured generation failed: {e}")
            full_prompt = f"{system_prompt}\n\n{prompt}" if system_prompt else prompt
            response_text = await self.generate(full_prompt)

            try:
                print("Falling back to manual JSON parsing")
                return self.parse_json_response(response_text, schema)

            except Exception as parse_error:
                raise RuntimeError(
                    f"Structured validation failed: {parse_error}"
                ) from e

    async def describe_image(
        self,
        prompt: str,
        image_path: str | Path,
        system_prompt: str = "",
        **kwargs: Any,
    ) -> str:
        """Generate a description of an image without blocking the event loop.

        The image is read on a thread and encoded as a data URL for the
        Gemini client.
        """
        debug_prompt = prompt[:100] + "..." if len(prompt) > 100 else prompt
        print(f"Generating text with prompt: {debug_prompt}")

        image_path = Path(image_path)

        # Run file I/O in a thread to avoid blocking the event loop.
        def read_image_file() -> bytes:
            with open(image_path, "rb") as f:
                return f.read()

        image_bytes = await asyncio.to_thread(read_image_file)
        base64_image = base64.b64encode(image_bytes).decode("utf-8")

        mime_type, _ = mimetypes.guess_type(str(image_path))
        mime_type = mime_type or "image/png"

        messages = []
        if system_prompt:
            messages.append(
                SystemMessage(content=[{"type": "text", "text": system_prompt}])
            )

        messages.append(
            HumanMessage(
                content=[
                    {"type": "text", "text": prompt},
                    {
                        "type": "image_url",
                        "image_url": {"url": f"data:{mime_type};base64,{base64_image}"},
                    },
                ]
            )
        )

        response = await self.llm.ainvoke(messages)
        content = getattr(response, "content", str(response))

        return str(content)
