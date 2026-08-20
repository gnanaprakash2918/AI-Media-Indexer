"""Concrete LLM/VLM provider implementations."""

import asyncio
import base64
from pathlib import Path
from typing import Any

import httpx

from config import settings
from core.llm.client import LLMClient, T
from core.utils.logger import log


class VLLMClient(LLMClient):
    """OpenAI-compatible provider backed by vLLM (or any OpenAI-compat endpoint)."""

    def __init__(self, base_url: str | None = None, model: str | None = None, api_key: str | None = None, prompt_dir: str | Path | None = None, timeout: float = 120.0):
        super().__init__(prompt_dir=prompt_dir)
        url = (base_url or settings.vllm_base_url).rstrip("/")
        if url.endswith("/v1"):
            url = url[:-3]
        self.base_url = url
        self.model = model or settings.vlm_endpoint_model_name
        self.api_key = api_key or settings.vllm_api_key
        self.timeout = timeout
        log(f"[VLLMClient] endpoint={self.base_url}  model={self.model}")

    def _headers(self) -> dict[str, str]:
        h: dict[str, str] = {"Content-Type": "application/json"}
        if self.api_key:
            h["Authorization"] = f"Bearer {self.api_key}"
        return h

    async def _chat_completion(self, messages: list[dict], max_tokens: int = 512, temperature: float = 0.0, response_format: dict | None = None, **kwargs: Any) -> str:
        payload: dict[str, Any] = {
            "model": self.model,
            "messages": messages,
            "max_tokens": max_tokens,
            "temperature": temperature,
        }
        if response_format:
            payload["response_format"] = response_format

        async with httpx.AsyncClient(timeout=self.timeout) as client:
            try:
                resp = await client.post(
                    f"{self.base_url}/v1/chat/completions",
                    headers=self._headers(),
                    json=payload,
                )
                resp.raise_for_status()
            except Exception as exc:
                log(f"[VLLMClient] Request failed: {exc}")
                raise

        data = resp.json()
        return data["choices"][0]["message"]["content"]

    async def generate(self, prompt: str, **kwargs: Any) -> str:
        return await self._chat_completion([{"role": "user", "content": prompt}], **kwargs)

    async def generate_structured(self, schema: type[T], prompt: str, system_prompt: str = "", **kwargs: Any) -> T:
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})

        try:
            text = await self._chat_completion(messages, response_format={"type": "json_object"}, **kwargs)
            return self.parse_json_response(text, schema)
        except Exception:
            text = await self._chat_completion(messages, **kwargs)
            return self.parse_json_response(text, schema)

    async def describe_image_from_bytes(self, prompt: str, image_bytes: bytes, system_prompt: str = "", **kwargs: Any) -> str:
        def _prepare_b64() -> str:
            import io
            from PIL import Image
            with Image.open(io.BytesIO(image_bytes)) as img:
                img = img.convert("RGB")
                if max(img.width, img.height) > 768:
                    img.thumbnail((768, 768), Image.Resampling.LANCZOS)
                buf = io.BytesIO()
                img.save(buf, format="JPEG", quality=85)
                return base64.b64encode(buf.getvalue()).decode("utf-8")

        b64 = await asyncio.to_thread(_prepare_b64)
        mime = "image/jpeg"

        content: list[dict] = [
            {"type": "text", "text": prompt},
            {"type": "image_url", "image_url": {"url": f"data:{mime};base64,{b64}"}},
        ]

        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": content})

        return await self._chat_completion(messages, **kwargs)

    async def describe_image(self, prompt: str, image_path: str | Path, system_prompt: str = "", **kwargs: Any) -> str:
        path = Path(image_path)
        if not path.exists():
            return ""
        with open(path, "rb") as f:
            return await self.describe_image_from_bytes(prompt, f.read(), system_prompt, **kwargs)


class OllamaClient(LLMClient):
    """Local Ollama provider (dev convenience)."""

    def __init__(self, base_url: str | None = None, model: str | None = None, prompt_dir: str | Path | None = None, timeout: float = 120.0):
        super().__init__(prompt_dir=prompt_dir)
        self.base_url = (base_url or settings.ollama_base_url).rstrip("/")
        self.model = model or settings.ollama_text_model
        self.vision_model = settings.ollama_vision_model
        self.timeout = timeout
        log(f"[OllamaClient] base_url={self.base_url} text_model={self.model} vision_model={self.vision_model}")

    async def _post(self, payload: dict) -> str:
        async with httpx.AsyncClient(timeout=self.timeout) as client:
            resp = await client.post(f"{self.base_url}/api/generate", json=payload)
            resp.raise_for_status()
            return resp.json().get("response", "").strip()

    async def generate(self, prompt: str, **kwargs: Any) -> str:
        return await self._post({"model": self.model, "prompt": prompt, "stream": False})

    async def generate_structured(self, schema: type[T], prompt: str, system_prompt: str = "", **kwargs: Any) -> T:
        full_prompt = f"{system_prompt}\n\n{prompt}" if system_prompt else prompt
        text = await self._post({"model": self.model, "prompt": full_prompt, "stream": False, "format": "json"})
        return self.parse_json_response(text, schema)

    async def describe_image_from_bytes(self, prompt: str, image_bytes: bytes, system_prompt: str = "", **kwargs: Any) -> str:
        b64 = base64.b64encode(image_bytes).decode("utf-8")
        full_prompt = f"{system_prompt}\n\n{prompt}" if system_prompt else prompt
        return await self._post({"model": self.vision_model, "prompt": full_prompt, "images": [b64], "stream": False})

    async def describe_image(self, prompt: str, image_path: str | Path, system_prompt: str = "", **kwargs: Any) -> str:
        path = Path(image_path)
        if not path.exists():
            return ""
        with open(path, "rb") as f:
            return await self.describe_image_from_bytes(prompt, f.read(), system_prompt, **kwargs)


class GeminiClient(LLMClient):
    """Google Gemini provider via langchain_google_genai."""

    def __init__(self, model: str | None = None, api_key: str | None = None, prompt_dir: str | Path | None = None):
        super().__init__(prompt_dir=prompt_dir)
        self.model = model or settings.gemini_model
        self.api_key = api_key or (settings.gemini_api_key.get_secret_value() if settings.gemini_api_key else None)
        self._llm = None

    def _get_llm(self):
        if self._llm is None:
            if not self.api_key:
                raise ValueError("GOOGLE_API_KEY not set for GeminiClient")
            from langchain_google_genai import ChatGoogleGenerativeAI
            self._llm = ChatGoogleGenerativeAI(model=self.model, api_key=self.api_key, temperature=0.0)
        return self._llm

    async def generate(self, prompt: str, **kwargs: Any) -> str:
        llm = self._get_llm()
        result = await llm.ainvoke(prompt)
        return str(getattr(result, "content", result)).strip()

    async def generate_structured(self, schema: type[T], prompt: str, system_prompt: str = "", **kwargs: Any) -> T:
        full_prompt = f"{system_prompt}\n\n{prompt}" if system_prompt else prompt
        text = await self.generate(full_prompt)
        return self.parse_json_response(text, schema)

    async def describe_image_from_bytes(self, prompt: str, image_bytes: bytes, system_prompt: str = "", **kwargs: Any) -> str:
        from langchain_core.messages import HumanMessage
        llm = self._get_llm()
        b64 = base64.b64encode(image_bytes).decode("utf-8")
        msg = HumanMessage(content=[
            {"type": "text", "text": f"{system_prompt}\n\n{prompt}" if system_prompt else prompt},
            {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{b64}"}}
        ])
        result = await llm.ainvoke([msg])
        return str(getattr(result, "content", result)).strip()

    async def describe_image(self, prompt: str, image_path: str | Path, system_prompt: str = "", **kwargs: Any) -> str:
        path = Path(image_path)
        if not path.exists():
            return ""
        with open(path, "rb") as f:
            return await self.describe_image_from_bytes(prompt, f.read(), system_prompt, **kwargs)


def get_client(provider: str | None = None) -> LLMClient:
    """Factory method to get the correct LLM implementation."""
    p = (provider or settings.llm_provider).lower()
    
    if p == "vllm":
        return VLLMClient()
    elif p == "gemini":
        return GeminiClient()
    elif p == "ollama":
        return OllamaClient()
    else:
        log(f"[LLMFactory] Unknown provider '{p}', defaulting to vllm")
        return VLLMClient()
