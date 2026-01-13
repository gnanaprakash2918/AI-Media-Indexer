"""Vision-Language Model (VLM) client factory and base classes."""

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
    """Abstract base class for VLM clients."""

    @abstractmethod
    def generate_caption(self, image_path: Path | str, prompt: str) -> str:
        """Generates a caption for an image file.

        Args:
            image_path: Path to the image file.
            prompt: The text prompt.

        Returns:
            The generated caption.
        """
        pass

    @abstractmethod
    def generate_caption_from_bytes(
        self, image_bytes: bytes, prompt: str
    ) -> str:
        """Generates a caption from image bytes.

        Args:
            image_bytes: The raw image bytes.
            prompt: The text prompt.

        Returns:
            The generated caption.
        """
        pass


class OllamaVLM(VLMClient):
    """Client for generating vision-language model (VLM) captions using Ollama."""

    def __init__(
        self,
        model: str | None = None,
        base_url: str | None = None,
        timeout: float = 60.0,
    ):
        """Initializes the Ollama VLM client.

        Args:
            model: Optional model name.
            base_url: Optional base URL for Ollama API.
            timeout: Request timeout in seconds.
        """
        self.model = model or settings.ollama_vision_model
        self.base_url = (base_url or settings.ollama_base_url).rstrip("/")
        self.timeout = timeout

    def generate_caption(self, image_path: Path | str, prompt: str) -> str:
        """Sends a file-based captioning request to Ollama."""
        path = Path(image_path)
        if not path.exists():
            return ""
        with open(path, "rb") as f:
            return self.generate_caption_from_bytes(f.read(), prompt)

    def generate_caption_from_bytes(
        self, image_bytes: bytes, prompt: str
    ) -> str:
        """Sends an image-bytes captioning request to Ollama."""
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
                resp = client.post(
                    f"{self.base_url}/api/generate", json=payload
                )
                resp.raise_for_status()
                return resp.json().get("response", "").strip()
        except Exception as e:
            log(f"OllamaVLM error: {e}")
            return ""


class GeminiVLM(VLMClient):
    """Client for generating vision-language model (VLM) captions using Google Gemini."""

    def __init__(self, model: str | None = None, api_key: str | None = None):
        """Initializes the Gemini VLM client.

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
        """Retrieves or initializes the Gemini GenerativeModel client."""
        if self._client is None:
            try:
                import google.generativeai as genai  # type: ignore

                if not self.api_key:
                    raise ValueError("GOOGLE_API_KEY not set")
                genai.configure(api_key=self.api_key)  # type: ignore
                self._client = genai.GenerativeModel(self.model)  # type: ignore
            except ImportError as e:
                raise ImportError("google-generativeai not installed") from e
        return self._client

    def generate_caption(self, image_path: Path | str, prompt: str) -> str:
        """Sends a file-based captioning request to Gemini."""
        from PIL import Image as PILImage

        path = Path(image_path)
        if not path.exists():
            return ""
        img = PILImage.open(path)
        return self._generate(img, prompt)

    def generate_caption_from_bytes(
        self, image_bytes: bytes, prompt: str
    ) -> str:
        """Sends an image-bytes captioning request to Gemini."""
        import io

        from PIL import Image as PILImage

        img = PILImage.open(io.BytesIO(image_bytes))
        return self._generate(img, prompt)

    def _generate(self, img, prompt: str) -> str:
        """Internal helper to generate content from an image and prompt using Gemini."""
        try:
            client = self._get_client()
            response = client.generate_content([prompt, img])
            return response.text.strip() if response.text else ""
        except Exception as e:
            log(f"GeminiVLM error: {e}")
            return ""


def get_vlm_client(provider: str | None = None) -> VLMClient:
    """Retrieves a VLM client based on the configured provider.

    Args:
        provider: Optional provider name ('ollama' or 'gemini').

    Returns:
        The initialized VLM client.
    """
    provider = provider or settings.ai_provider_vision
    if provider == "gemini":
        return GeminiVLM()
    return OllamaVLM()
