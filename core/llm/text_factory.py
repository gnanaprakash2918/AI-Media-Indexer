from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Type, TypeVar

from pydantic import BaseModel

from config import settings
from core.utils.logger import log

T = TypeVar("T", bound=BaseModel)


class TextLLMClient(ABC):
    @abstractmethod
    def generate(self, prompt: str) -> str:
        pass

    def generate_json(self, prompt: str, schema: Type[T]) -> T | None:
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
    def __init__(
        self,
        model: str | None = None,
        base_url: str | None = None,
        timeout: float = 60.0,
    ):
        self.model = model or settings.ollama_model
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
                resp = client.post(f"{self.base_url}/api/generate", json=payload)
                resp.raise_for_status()
                return resp.json().get("response", "").strip()
        except Exception as e:
            log(f"OllamaText error: {e}")
            return ""


class GeminiText(TextLLMClient):
    def __init__(self, model: str | None = None, api_key: str | None = None):
        self.model = model or settings.gemini_model
        self.api_key = api_key or (
            settings.gemini_api_key.get_secret_value()
            if settings.gemini_api_key
            else None
        )
        self._client = None

    def _get_client(self):
        if self._client is None:
            try:
                import google.generativeai as genai  # type: ignore

                if not self.api_key:
                    raise ValueError("GOOGLE_API_KEY not set")
                genai.configure(api_key=self.api_key)  # type: ignore
                self._client = genai.GenerativeModel(  # type: ignore
                    self.model,
                    generation_config={"response_mime_type": "application/json"},
                )
            except ImportError:
                raise ImportError("google-generativeai not installed")
        return self._client

    def generate(self, prompt: str) -> str:
        try:
            client = self._get_client()
            response = client.generate_content(prompt)
            return response.text.strip() if response.text else ""
        except Exception as e:
            log(f"GeminiText error: {e}")
            return ""


def get_text_client(provider: str | None = None) -> TextLLMClient:
    provider = provider or settings.ai_provider_text
    if provider == "gemini":
        return GeminiText()
    return OllamaText()
