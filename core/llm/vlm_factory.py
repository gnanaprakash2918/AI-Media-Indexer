"""Vision-Language Model (VLM) client factory.

All providers implement VLMClient (synchronous frame captioning interface).
Switch providers via settings.ai_provider_vision or the ai_provider_vision
env var — no code changes at call sites.

Supported providers:
    vllm   — Qwen3-VL via vLLM OpenAI-compat endpoint (DEFAULT)
    ollama — Local Ollama vision model (dev convenience)
    gemini — Google Gemini (cloud; uses langchain_google_genai, not bare google.generativeai)

NOTE: google.generativeai is NOT imported directly anywhere in this file.
All Gemini calls go through langchain_google_genai to maintain provider isolation.
"""

from __future__ import annotations

import base64
from abc import ABC, abstractmethod
from pathlib import Path

from config import settings
from core.utils.logger import log


class VLMClient(ABC):
    """Abstract base class for synchronous VLM frame-captioning clients.

    Used by scene captioning (SceneStageMixin) and reranking (RerankingCouncil)
    for single-frame or multi-frame image description.
    """

    @abstractmethod
    def generate_caption(self, image_path: Path | str, prompt: str) -> str:
        """Generate a caption for an image file.

        Args:
            image_path: Path to the image file.
            prompt:     The text prompt.

        Returns:
            The generated caption string.
        """

    @abstractmethod
    def generate_caption_from_bytes(
        self, image_bytes: bytes, prompt: str
    ) -> str:
        """Generate a caption from raw image bytes.

        Args:
            image_bytes: Raw image bytes (JPEG, PNG, etc.).
            prompt:      The text prompt.

        Returns:
            The generated caption string.
        """


class VLLMVLMClient(VLMClient):
    """VLM client that calls a vLLM (or any OpenAI-compat) vision endpoint.

    Sends images as base64 data-URIs in the OpenAI image_url content format.
    Works with Qwen3-VL, LLaVA, and any vision model served by vLLM.
    """

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
        log(
            f"[VLLMVLMClient] endpoint={self.base_url}  model={self.model}"
        )

    def _headers(self) -> dict[str, str]:
        h = {"Content-Type": "application/json"}
        if self.api_key:
            h["Authorization"] = f"Bearer {self.api_key}"
        return h

    def _post(self, image_bytes: bytes, prompt: str) -> str:
        import httpx

        b64 = base64.b64encode(image_bytes).decode("utf-8")
        # Detect MIME type from magic bytes (JPEG / PNG / WebP / GIF)
        if image_bytes[:2] == b"\xff\xd8":
            mime = "image/jpeg"
        elif image_bytes[:8] == b"\x89PNG\r\n\x1a\n":
            mime = "image/png"
        else:
            mime = "image/jpeg"  # safe default for cv2-encoded frames

        payload = {
            "model": self.model,
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt},
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:{mime};base64,{b64}"
                            },
                        },
                    ],
                }
            ],
            "max_tokens": 256,
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
                return resp.json()["choices"][0]["message"]["content"].strip()
        except httpx.ConnectError as exc:
            log(
                f"[VLLMVLMClient] Cannot connect to {self.base_url}. "
                f"Set LLM_PROVIDER=ollama for dev without vLLM. Error: {exc}"
            )
            return ""
        except Exception as exc:
            log(f"[VLLMVLMClient] Error: {exc}")
            return ""

    def generate_caption(self, image_path: Path | str, prompt: str) -> str:
        path = Path(image_path)
        if not path.exists():
            return ""
        with open(path, "rb") as f:
            return self.generate_caption_from_bytes(f.read(), prompt)

    def generate_caption_from_bytes(
        self, image_bytes: bytes, prompt: str
    ) -> str:
        return self._post(image_bytes, prompt)


class OllamaVLM(VLMClient):
    """VLM client backed by local Ollama (dev convenience)."""

    def __init__(
        self,
        model: str | None = None,
        base_url: str | None = None,
        timeout: float = 60.0,
    ):
        self.model = model or settings.ollama_vision_model
        self.base_url = (base_url or settings.ollama_base_url).rstrip("/")
        self.timeout = timeout
        log(
            f"[OllamaVLM] Initialized with model={self.model}, base_url={self.base_url}"
        )

    def generate_caption(self, image_path: Path | str, prompt: str) -> str:
        path = Path(image_path)
        if not path.exists():
            return ""
        with open(path, "rb") as f:
            return self.generate_caption_from_bytes(f.read(), prompt)

    def generate_caption_from_bytes(
        self, image_bytes: bytes, prompt: str
    ) -> str:
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
            log(f"[OllamaVLM] error: {e}")
            return ""


class GeminiVLM(VLMClient):
    """VLM client backed by Google Gemini (cloud).

    Uses langchain_google_genai — NOT the bare google.generativeai SDK.
    This maintains provider isolation: Gemini is never imported directly
    at call sites.
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
                log("[GeminiVLM] GOOGLE_API_KEY not set, Gemini disabled")
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
                    "[GeminiVLM] langchain_google_genai not installed, Gemini disabled"
                )
                return None
            except Exception as e:
                log(f"[GeminiVLM] Failed to initialize: {e}")
                return None
        return self._llm

    def _generate(self, image_bytes: bytes, prompt: str) -> str:
        import asyncio
        import base64 as _b64
        import io
        import mimetypes

        from langchain_core.messages import HumanMessage

        llm = self._get_llm()
        if llm is None:
            return ""

        b64 = _b64.b64encode(image_bytes).decode("utf-8")
        # Detect MIME
        try:
            from PIL import Image as PILImage

            img = PILImage.open(io.BytesIO(image_bytes))
            fmt = (img.format or "JPEG").lower()
            mime = f"image/{fmt}"
        except Exception:
            mime = "image/jpeg"

        msg = HumanMessage(
            content=[
                {"type": "text", "text": prompt},
                {
                    "type": "image_url",
                    "image_url": {"url": f"data:{mime};base64,{b64}"},
                },
            ]
        )
        try:
            # ChatGoogleGenerativeAI is async-capable; run sync wrapper
            result = asyncio.get_event_loop().run_until_complete(
                llm.ainvoke([msg])
            )
            content = getattr(result, "content", str(result))
            return str(content).strip()
        except Exception as e:
            log(f"[GeminiVLM] error: {e}")
            return ""

    def generate_caption(self, image_path: Path | str, prompt: str) -> str:
        path = Path(image_path)
        if not path.exists():
            return ""
        with open(path, "rb") as f:
            return self.generate_caption_from_bytes(f.read(), prompt)

    def generate_caption_from_bytes(
        self, image_bytes: bytes, prompt: str
    ) -> str:
        return self._generate(image_bytes, prompt)


def get_vlm_client(provider: str | None = None) -> VLMClient:
    """Get a VLM client for the configured provider.

    Args:
        provider: "vllm" | "ollama" | "gemini". Falls back to
                  settings.ai_provider_vision (default: "vllm").

    Returns:
        Initialized VLMClient implementation.
    """
    p = (provider or settings.ai_provider_vision).lower()

    if p == "vllm":
        return VLLMVLMClient()
    elif p == "gemini":
        return GeminiVLM()
    elif p == "ollama":
        return OllamaVLM()
    else:
        log(
            f"[get_vlm_client] Unknown provider '{p}', defaulting to vllm"
        )
        return VLLMVLMClient()
