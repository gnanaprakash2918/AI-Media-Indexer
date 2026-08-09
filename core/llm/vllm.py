"""VLLMProvider — thin OpenAI-compatible HTTP client implementing LLMInterface.

Works with any OpenAI-compatible endpoint:
  - Local vLLM instance serving Qwen3-VL or any instruct model
  - Hosted endpoints: Together AI, Fireworks, Anyscale, etc.
  - Any self-hosted server that implements /v1/chat/completions

No local model loading. No GPU allocation. Pure HTTP.

Configuration (all via env vars or Settings):
    VLLM_BASE_URL   — base URL of the endpoint (default: http://localhost:8000)
    VLLM_API_KEY    — bearer token if required (default: None)
    VLM_ENDPOINT_MODEL_NAME — model name sent to the endpoint
"""

from __future__ import annotations

import asyncio
import base64
import mimetypes
from pathlib import Path
from typing import Any

import httpx
from pydantic import BaseModel

from .interface import LLMInterface, T


class VLLMProvider(LLMInterface):
    """OpenAI-compatible provider backed by vLLM (or any OpenAI-compat endpoint).

    Implements the full LLMInterface ABC:
      - generate()           → /v1/chat/completions (text)
      - generate_structured() → /v1/chat/completions + JSON-mode / schema parsing
      - describe_image()     → /v1/chat/completions with base64 image content
      - unload_model()       → no-op (remote; nothing to unload locally)
    """

    def __init__(
        self,
        base_url: str | None = None,
        model: str | None = None,
        api_key: str | None = None,
        prompt_dir: str | Path | None = None,
        timeout: float = 120.0,
    ):
        """Initialize the vLLM provider.

        Args:
            base_url:   vLLM endpoint base URL. Falls back to settings.vllm_base_url.
            model:      Model name to pass to the endpoint. Falls back to
                        settings.vlm_endpoint_model_name.
            api_key:    Bearer token for auth (None = no auth header sent).
            prompt_dir: Directory for prompt templates.
            timeout:    HTTP request timeout in seconds.
        """
        from config import settings  # defer to avoid circular import at module level

        super().__init__(prompt_dir=prompt_dir)

        url = (base_url or settings.vllm_base_url).rstrip("/")
        if url.endswith("/v1"):
            url = url[:-3]
        self.base_url = url
        self.model = model or settings.vlm_endpoint_model_name
        self.api_key = api_key or settings.vllm_api_key
        self.timeout = timeout

        print(
            f"[VLLMProvider] endpoint={self.base_url}  model={self.model}"
        )

    # ------------------------------------------------------------------
    # Internal helper
    # ------------------------------------------------------------------

    def _headers(self) -> dict[str, str]:
        h: dict[str, str] = {"Content-Type": "application/json"}
        if self.api_key:
            h["Authorization"] = f"Bearer {self.api_key}"
        return h

    async def _chat_completion(
        self,
        messages: list[dict],
        max_tokens: int = 512,
        temperature: float = 0.0,
        response_format: dict | None = None,
        **kwargs: Any,
    ) -> str:
        """Send a /v1/chat/completions request and return the assistant text."""
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
            except httpx.ConnectError as exc:
                raise RuntimeError(
                    f"[VLLMProvider] Cannot connect to vLLM at {self.base_url}. "
                    f"Make sure vLLM is running, or set LLM_PROVIDER=ollama for "
                    f"local dev without a GPU/vLLM instance. "
                    f"Original error: {exc}"
                ) from exc
            except httpx.HTTPStatusError as exc:
                error_detail = resp.text
                try:
                    err_json = resp.json()
                    if isinstance(err_json, dict) and "message" in err_json.get("error", {}):
                        error_detail = err_json["error"]["message"]
                    elif isinstance(err_json, dict) and "message" in err_json:
                        error_detail = err_json["message"]
                except Exception:
                    pass
                raise RuntimeError(
                    f"[VLLMProvider] vLLM endpoint returned HTTP {resp.status_code}: {error_detail}"
                ) from exc

        data = resp.json()
        return data["choices"][0]["message"]["content"]

    # ------------------------------------------------------------------
    # LLMInterface implementation
    # ------------------------------------------------------------------

    async def generate(self, prompt: str, **kwargs: Any) -> str:
        """Generate a text response from a plain prompt."""
        messages = [{"role": "user", "content": prompt}]
        return await self._chat_completion(messages, **kwargs)

    async def generate_structured(
        self,
        schema: type[T],
        prompt: str,
        system_prompt: str = "",
        **kwargs: Any,
    ) -> T:
        """Generate a structured response validated against a Pydantic schema.

        Tries JSON-mode first (/v1 response_format=json_object).
        Falls back to text generation + manual JSON parsing on failure.
        """
        messages: list[dict] = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})

        # Try JSON mode first (vLLM supports this for most models)
        try:
            text = await self._chat_completion(
                messages,
                response_format={"type": "json_object"},
                **kwargs,
            )
            return self.parse_json_response(text, schema)
        except Exception:
            pass

        # Fallback: plain text + JSON parsing
        text = await self._chat_completion(messages, **kwargs)
        return self.parse_json_response(text, schema)

    async def describe_image(
        self,
        prompt: str,
        image_path: str | Path,
        system_prompt: str = "",
        **kwargs: Any,
    ) -> str:
        """Describe an image using the vLLM vision endpoint.

        The image is auto-scaled (max dimension 768px) and base64-encoded
        as an OpenAI-style image_url content part to ensure token count
        remains low (~400 tokens) and fits within context bounds.
        """
        image_path = Path(image_path)

        def _prepare_b64() -> str:
            import io
            from PIL import Image

            with Image.open(image_path) as img:
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
            {
                "type": "image_url",
                "image_url": {"url": f"data:{mime};base64,{b64}"},
            },
        ]

        messages: list[dict] = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": content})

        return await self._chat_completion(messages, **kwargs)

    async def unload_model(self) -> None:
        """No-op — remote endpoint; nothing to unload locally."""
        pass
