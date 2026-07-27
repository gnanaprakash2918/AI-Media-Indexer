"""Text LLM client factory.

All providers implement TextLLMClient (synchronous text generation interface).
Switch providers via settings.ai_provider_text or the ai_provider_text env var.

Supported providers:
    vllm   — Any OpenAI-compatible endpoint (DEFAULT)
    ollama — Local Ollama (dev convenience)
    gemini — Google Gemini (cloud; uses langchain_google_genai)

NOTE: google.generativeai is NOT imported directly in this file.
All Gemini calls go through langchain_google_genai to maintain provider isolation.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TypeVar

from pydantic import BaseModel

from config import settings
from core.utils.logger import log

T = TypeVar("T", bound=BaseModel)


class TextLLMClient(ABC):
    """Abstract base class for synchronous text LLM clients."""

    @abstractmethod
    def generate(self, prompt: str) -> str:
        """Generate a text response from the given prompt."""

    def generate_json(self, prompt: str, schema: type[T]) -> T | None:
        """Generate a JSON response matching the provided schema.

        Args:
            prompt: The text prompt.
            schema: Pydantic model class to validate against.

        Returns:
            An instance of the schema, or None if generation/parsing fails.
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
            log(f"[TextLLMClient] JSON parse failed: {e}")
            try:
                start = raw.find("{")
                end = raw.rfind("}") + 1
                if start >= 0 and end > start:
                    return schema.model_validate_json(raw[start:end])
            except Exception:
                pass
            return None


class VLLMText(TextLLMClient):
    """Text client that calls a vLLM (or any OpenAI-compat) endpoint."""

    def __init__(
        self,
        base_url: str | None = None,
        model: str | None = None,
        api_key: str | None = None,
        timeout: float = 60.0,
    ):
        self.base_url = (base_url or settings.vllm_base_url).rstrip("/")
        self.model = model or settings.vlm_endpoint_model_name
        self.api_key = api_key or settings.vllm_api_key
        self.timeout = timeout

    def _headers(self) -> dict[str, str]:
        h = {"Content-Type": "application/json"}
        if self.api_key:
            h["Authorization"] = f"Bearer {self.api_key}"
        return h

    def generate(self, prompt: str) -> str:
        import httpx

        payload = {
            "model": self.model,
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": 512,
            "temperature": 0.0,
        }
        try:
            with httpx.Client(timeout=self.timeout) as client:
                resp = client.post(
                    f"{self.base_url}/v1/chat/completions",
                    headers=self._headers(),
                    json=payload,
                )
                resp.raise_for_status()
                return (
                    resp.json()["choices"][0]["message"]["content"].strip()
                )
        except httpx.ConnectError as exc:
            log(
                f"[VLLMText] Cannot connect to {self.base_url}. "
                f"Set LLM_PROVIDER=ollama for dev without vLLM. Error: {exc}"
            )
            return ""
        except Exception as e:
            log(f"[VLLMText] error: {e}")
            return ""


class OllamaText(TextLLMClient):
    """Text client backed by local Ollama (dev convenience)."""

    def __init__(
        self,
        model: str | None = None,
        base_url: str | None = None,
        timeout: float = 60.0,
    ):
        self.model = model or settings.ollama_text_model
        self.base_url = (base_url or settings.ollama_base_url).rstrip("/")
        self.timeout = timeout

    def generate(self, prompt: str) -> str:
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
            log(f"[OllamaText] error: {e}")
            return ""


class GeminiText(TextLLMClient):
    """Text client backed by Google Gemini (cloud).

    Uses langchain_google_genai — NOT the bare google.generativeai SDK.
    """

    def __init__(self, model: str | None = None, api_key: str | None = None):
        self.model = model or settings.gemini_model
        self.api_key = api_key or (
            settings.gemini_api_key.get_secret_value()
            if settings.gemini_api_key
            else None
        )
        self._llm = None

    def _get_llm(self):
        """Lazily initialize langchain_google_genai ChatGoogleGenerativeAI."""
        if self._llm is None:
            if not self.api_key:
                log("[GeminiText] GOOGLE_API_KEY not set, Gemini disabled")
                return None
            try:
                from langchain_google_genai import ChatGoogleGenerativeAI

                self._llm = ChatGoogleGenerativeAI(
                    model=self.model,
                    api_key=self.api_key,
                    temperature=0.0,
                )
            except ImportError:
                log(
                    "[GeminiText] langchain_google_genai not installed, Gemini disabled"
                )
                return None
            except Exception as e:
                log(f"[GeminiText] Failed to initialize: {e}")
                return None
        return self._llm

    def generate(self, prompt: str) -> str:
        import asyncio

        llm = self._get_llm()
        if llm is None:
            return ""
        try:
            result = asyncio.get_event_loop().run_until_complete(
                llm.ainvoke(prompt)
            )
            content = getattr(result, "content", str(result))
            return str(content).strip()
        except Exception as e:
            log(f"[GeminiText] error: {e}")
            return ""


def get_text_client(provider: str | None = None) -> TextLLMClient:
    """Get a text LLM client for the configured provider.

    Args:
        provider: "vllm" | "ollama" | "gemini". Falls back to
                  settings.ai_provider_text (default: "vllm").

    Returns:
        Initialized TextLLMClient implementation.
    """
    p = (provider or settings.ai_provider_text).lower()

    if p == "vllm":
        return VLLMText()
    elif p == "gemini":
        return GeminiText()
    elif p == "ollama":
        return OllamaText()
    else:
        log(
            f"[get_text_client] Unknown provider '{p}', defaulting to vllm"
        )
        return VLLMText()
