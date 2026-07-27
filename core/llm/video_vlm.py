"""Video-Native VLM Client for Qwen3-VL via vLLM.

Replaces the previous local HuggingFace model load (Qwen2-VL) with a
thin HTTP client calling the vLLM OpenAI-compatible /v1/chat/completions
endpoint. No local GPU allocation, no transformers import, no weights on disk.

The endpoint is configured via:
    VLLM_BASE_URL          — base URL (default: http://localhost:8000)
    VLM_ENDPOINT_MODEL_NAME — model name served by vLLM (Qwen/Qwen3-VL-2B-Instruct)
    VLLM_API_KEY           — bearer token if required (default: None)

For local dev without vLLM, set LLM_PROVIDER=ollama — VideoVLM will use
the Ollama vision model as a fallback.
"""

from __future__ import annotations

import asyncio
import base64
from typing import Any

import numpy as np

from config import settings
from core.utils.logger import get_logger

log = get_logger(__name__)


class VideoVLM:
    """Video Understanding VLM client (Qwen3-VL via vLLM endpoint).

    Sends frame sequences to the vLLM /v1/chat/completions endpoint
    formatted as OpenAI-style image_url content parts. No local model
    loading — all inference happens on the vLLM server.
    """

    def __init__(
        self,
        base_url: str | None = None,
        model: str | None = None,
        api_key: str | None = None,
        timeout: float = 120.0,
    ):
        """Initialize the VideoVLM client.

        Args:
            base_url: vLLM endpoint base URL. Defaults to settings.vllm_base_url.
            model:    Model name sent to the endpoint. Defaults to
                      settings.vlm_endpoint_model_name.
            api_key:  Bearer token. Defaults to settings.vllm_api_key.
            timeout:  HTTP request timeout in seconds.
        """
        self.base_url = (base_url or settings.vllm_base_url).rstrip("/")
        self.model = model or settings.vlm_endpoint_model_name
        self.api_key = api_key or settings.vllm_api_key
        self.timeout = timeout

        log.info(
            f"[VideoVLM] endpoint={self.base_url}  model={self.model}"
        )

    def _headers(self) -> dict[str, str]:
        h = {"Content-Type": "application/json"}
        if self.api_key:
            h["Authorization"] = f"Bearer {self.api_key}"
        return h

    @staticmethod
    def _encode_frame(frame: np.ndarray) -> str:
        """Encode an RGB numpy frame to a base64 JPEG string."""
        import cv2

        # frame is RGB from PIL/numpy — convert to BGR for cv2.imencode
        bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        ok, buf = cv2.imencode(".jpg", bgr, [cv2.IMWRITE_JPEG_QUALITY, 80])
        if not ok:
            raise ValueError("Failed to encode frame as JPEG")
        return base64.b64encode(buf.tobytes()).decode("utf-8")

    async def generate_action_summary(
        self, frames: list[np.ndarray], fps: float = 1.0
    ) -> dict[str, str]:
        """Generate a structured action summary from video frames.

        Samples up to vlm_max_frames frames, encodes them as base64 JPEG,
        and sends them to the Qwen3-VL endpoint in a single chat request.

        Args:
            frames: List of RGB numpy frames (H, W, 3).
            fps:    Approximate frame rate of the list (informational only).

        Returns:
            Dict with keys 'action', 'subject', 'mood', and 'raw'.
            Returns empty dict on connection/inference error.
        """
        if not frames:
            return {}

        try:
            import httpx

            # --- Sample frames ---
            max_frames = settings.vlm_max_frames
            sampled = frames
            if len(frames) > max_frames:
                indices = np.linspace(
                    0, len(frames) - 1, max_frames, dtype=int
                )
                sampled = [frames[i] for i in indices]

            # --- Build message content ---
            # Encode frames as base64 images in a thread (CPU-bound)
            def _encode_all() -> list[str]:
                return [self._encode_frame(f) for f in sampled]

            encoded_frames = await asyncio.to_thread(_encode_all)

            content: list[dict[str, Any]] = []
            for b64 in encoded_frames:
                content.append(
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/jpeg;base64,{b64}"
                        },
                    }
                )
            content.append(
                {
                    "type": "text",
                    "text": (
                        f"These are {len(sampled)} frames sampled from a video "
                        f"at approximately {fps:.1f} fps. "
                        "Analyze the clip and describe the main action, "
                        "the subjects involved, and the overall mood. "
                        "Ignore static poses; focus on movement and interaction. "
                        "Format your answer exactly as: "
                        "Action: <action> | Subjects: <subjects> | Mood: <mood>"
                    ),
                }
            )

            payload: dict[str, Any] = {
                "model": self.model,
                "messages": [{"role": "user", "content": content}],
                "max_tokens": settings.vlm_max_tokens,
                "temperature": 0.0,
            }

            # --- Send to vLLM ---
            async with httpx.AsyncClient(timeout=self.timeout) as client:
                try:
                    resp = await client.post(
                        f"{self.base_url}/v1/chat/completions",
                        headers=self._headers(),
                        json=payload,
                    )
                    resp.raise_for_status()
                except httpx.ConnectError as exc:
                    log.error(
                        f"[VideoVLM] Cannot connect to vLLM at {self.base_url}. "
                        f"Check VLLM_BASE_URL or set LLM_PROVIDER=ollama for "
                        f"dev without a GPU/vLLM instance. Error: {exc}"
                    )
                    return {}

            output_text = resp.json()["choices"][0]["message"]["content"]

            # --- Parse "Action: X | Subjects: Y | Mood: Z" ---
            parts = output_text.split("|")
            result: dict[str, str] = {"raw": output_text}
            for part in parts:
                if ":" in part:
                    k, v = part.split(":", 1)
                    result[k.strip().lower()] = v.strip()

            log.debug(f"[VideoVLM] Summary: {result}")
            return result

        except Exception as e:
            log.error(f"[VideoVLM] Generation failed: {e}")
            return {}

    def cleanup(self) -> None:
        """No-op — remote endpoint; nothing to unload locally."""
        pass
