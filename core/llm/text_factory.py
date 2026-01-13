"""Text LLM client factory and base classes."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TypeVar

from pydantic import BaseModel

from config import settings
from core.utils.logger import log

T = TypeVar("T", bound=BaseModel)


class TextLLMClient(ABC):
    """Abstract base class for Text LLM clients."""

    @abstractmethod
    def generate(self, prompt: str) -> str:
        """Generates a text response from the given prompt.

        Args:
            prompt: The text prompt.

        Returns:
            The generated text response.
        """
        pass

    def generate_json(self, prompt: str, schema: type[T]) -> T | None:
        """Generates a JSON response matching the provided schema.

        Args:
            prompt: The text prompt.
            schema: The Pydantic model class to validate against.

        Returns:
            An instance of the schema, or None if generation or parsing fails.
        """
        raw = self.generate(prompt)
        if not raw:
            return None
        try:
            clean = raw.strip()
            if clean.startswith("```"):
                clean = clean.split("```")[1]
                if clean.startswith("json"):
                    clean = clean[4:]
            return schema.model_validate_json(clean)
        except Exception as e:
            log(f"JSON parse failed: {e}")
            try:
                start = raw.find("{")
                end = raw.rfind("}") + 1
                if start >= 0 and end > start:
                    return schema.model_validate_json(raw[start:end])
            except Exception:
                pass
            return None


class OllamaText(TextLLMClient):
    """Client for generating text using Ollama."""

    def __init__(
        self,
        model: str | None = None,
        base_url: str | None = None,
        timeout: float = 60.0,
    ):
        """Initializes the Ollama text client.

        Args:
            model: Optional model name.
            base_url: Optional base URL for the Ollama API.
            timeout: Request timeout in seconds.
        """
        self.model = model or settings.ollama_model
        self.base_url = (base_url or settings.ollama_base_url).rstrip("/")
        self.timeout = timeout

    def generate(self, prompt: str) -> str:
        """Sends a text generation request to Ollama."""
        import httpx

        payload = {
            "model": self.model,
            "prompt": prompt,
            "stream": False,
            "format": "json",
        }
        try:
            with httpx.Client(timeout=self.timeout) as client:
                resp = client.post(
                    f"{self.base_url}/api/generate", json=payload
                )
                resp.raise_for_status()
                return resp.json().get("response", "").strip()
        except Exception as e:
            log(f"OllamaText error: {e}")
            return ""


class GeminiText(TextLLMClient):
    """Client for generating text using Google Gemini."""

    def __init__(self, model: str | None = None, api_key: str | None = None):
        """Initializes the Gemini text client.

        Args:
            model: Optional model name.
            api_key: Optional Gemini API key.
        """
        self.model = model or settings.gemini_model
        self.api_key = api_key or (
            settings.gemini_api_key.get_secret_value()
            if settings.gemini_api_key
            else None
        )
        self._client = None

    def _get_client(self):
        """Retrieves or initializes the Gemini generative model client."""
        if self._client is None:
            try:
                import google.generativeai as genai  # type: ignore

                if not self.api_key:
                    raise ValueError("GOOGLE_API_KEY not set")
                genai.configure(api_key=self.api_key)  # type: ignore
                self._client = genai.GenerativeModel(  # type: ignore
                    self.model,
                    generation_config={
                        "response_mime_type": "application/json"
                    },
                )
            except ImportError as e:
                raise ImportError("google-generativeai not installed") from e
        return self._client

    def generate(self, prompt: str) -> str:
        """Sends a content generation request to Gemini."""
        try:
            client = self._get_client()
            response = client.generate_content(prompt)
            return response.text.strip() if response.text else ""
        except Exception as e:
            log(f"GeminiText error: {e}")
            return ""


def get_text_client(provider: str | None = None) -> TextLLMClient:
    """Retrieves a text LLM client based on the configured provider.

    Args:
        provider: Optional provider name ('ollama' or 'gemini').

    Returns:
        The initialized text LLM client.
    """
    provider = provider or settings.ai_provider_text
    if provider == "gemini":
        return GeminiText()
    return OllamaText()
