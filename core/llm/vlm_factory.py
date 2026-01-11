from __future__ import annotations

import base64
from abc import ABC, abstractmethod
from pathlib import Path
from typing import TYPE_CHECKING

from config import settings
from core.utils.logger import log

if TYPE_CHECKING:
    pass


class VLMClient(ABC):
    @abstractmethod
    def generate_caption(self, image_path: Path | str, prompt: str) -> str:
        pass

    @abstractmethod
    def generate_caption_from_bytes(self, image_bytes: bytes, prompt: str) -> str:
        pass


class OllamaVLM(VLMClient):
    def __init__(self, model: str | None = None, base_url: str | None = None, timeout: float = 60.0):
        self.model = model or settings.ollama_vision_model
        self.base_url = (base_url or settings.ollama_base_url).rstrip("/")
        self.timeout = timeout

    def generate_caption(self, image_path: Path | str, prompt: str) -> str:
        path = Path(image_path)
        if not path.exists():
            return ""
        with open(path, "rb") as f:
            return self.generate_caption_from_bytes(f.read(), prompt)

    def generate_caption_from_bytes(self, image_bytes: bytes, prompt: str) -> str:
        import httpx
        b64 = base64.b64encode(image_bytes).decode("utf-8")
        payload = {
            "model": self.model,
            "prompt": prompt,
            "images": [b64],
            "stream": False,
        }
        try:
            with httpx.Client(timeout=self.timeout) as client:
                resp = client.post(f"{self.base_url}/api/generate", json=payload)
                resp.raise_for_status()
                return resp.json().get("response", "").strip()
        except Exception as e:
            log(f"OllamaVLM error: {e}")
            return ""


class GeminiVLM(VLMClient):
    def __init__(self, model: str | None = None, api_key: str | None = None):
        self.model = model or settings.gemini_model
        self.api_key = api_key or (settings.gemini_api_key.get_secret_value() if settings.gemini_api_key else None)
        self._client = None

    def _get_client(self):
        if self._client is None:
            try:
                import google.generativeai as genai
                if not self.api_key:
                    raise ValueError("GOOGLE_API_KEY not set")
                genai.configure(api_key=self.api_key) # type: ignore
                self._client = genai.GenerativeModel(self.model) # type: ignore
            except ImportError:
                raise ImportError("google-generativeai not installed")
        return self._client

    def generate_caption(self, image_path: Path | str, prompt: str) -> str:
        from PIL import Image as PILImage
        path = Path(image_path)
        if not path.exists():
            return ""
        img = PILImage.open(path)
        return self._generate(img, prompt)

    def generate_caption_from_bytes(self, image_bytes: bytes, prompt: str) -> str:
        import io

        from PIL import Image as PILImage
        img = PILImage.open(io.BytesIO(image_bytes))
        return self._generate(img, prompt)

    def _generate(self, img, prompt: str) -> str:
        try:
            client = self._get_client()
            response = client.generate_content([prompt, img])
            return response.text.strip() if response.text else ""
        except Exception as e:
            log(f"GeminiVLM error: {e}")
            return ""


def get_vlm_client(provider: str | None = None) -> VLMClient:
    provider = provider or settings.ai_provider_vision
    if provider == "gemini":
        return GeminiVLM()
    return OllamaVLM()
