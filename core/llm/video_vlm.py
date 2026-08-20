"""Video-Native VLM Client for Qwen3-VL via vLLM.

Replaces the previous local HuggingFace model load (Qwen2-VL) with a
thin wrapper around the unified LLMClient.
"""

from __future__ import annotations

import asyncio
import base64
from typing import Any

import numpy as np

from config import settings
from core.utils.logger import get_logger
from core.llm.providers import get_client, VLLMClient

log = get_logger(__name__)


class VideoVLM:
    """Video Understanding VLM client using the unified VLLMClient."""

    def __init__(self, base_url: str | None = None, model: str | None = None, api_key: str | None = None, timeout: float = 120.0):
        # Always forces VLLM Provider for multi-frame video understanding
        self.client = get_client("vllm")
        # Override settings if provided
        if isinstance(self.client, VLLMClient):
            if base_url:
                self.client.base_url = base_url.rstrip("/")
                if self.client.base_url.endswith("/v1"):
                    self.client.base_url = self.client.base_url[:-3]
            if model:
                self.client.model = model
            if api_key:
                self.client.api_key = api_key
            self.client.timeout = timeout

    @staticmethod
    def _encode_frame(frame: np.ndarray, max_dim: int = 512) -> str:
        """Encode an RGB numpy frame to a base64 JPEG string, capping max dimension."""
        import cv2

        h, w = frame.shape[:2]
        if max(h, w) > max_dim:
            scale = max_dim / float(max(h, w))
            new_w, new_h = max(1, int(w * scale)), max(1, int(h * scale))
            frame = cv2.resize(frame, (new_w, new_h), interpolation=cv2.INTER_AREA)

        bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        ok, buf = cv2.imencode(".jpg", bgr, [cv2.IMWRITE_JPEG_QUALITY, 80])
        if not ok:
            raise ValueError("Failed to encode frame as JPEG")
        return base64.b64encode(buf.tobytes()).decode("utf-8")

    async def generate_action_summary(self, frames: list[np.ndarray], fps: float = 1.0) -> dict[str, str]:
        if not frames:
            return {}

        try:
            max_frames = settings.vlm_max_frames
            sampled = frames
            if len(frames) > max_frames:
                indices = np.linspace(0, len(frames) - 1, max_frames, dtype=int)
                sampled = [frames[i] for i in indices]

            async def _do_request(frames_to_send: list[np.ndarray]) -> str:
                encoded_frames = await asyncio.to_thread(lambda: [self._encode_frame(f) for f in frames_to_send])

                content: list[dict[str, Any]] = []
                for b64 in encoded_frames:
                    content.append({"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{b64}"}})
                
                content.append({
                    "type": "text",
                    "text": (
                        f"These are {len(frames_to_send)} frames sampled from a video clip at approximately {fps:.1f} fps. "
                        "Analyze the clip and describe the main action, the subjects involved, and the overall mood. "
                        "Ignore static poses; focus on movement and interaction. "
                        "Format your answer exactly as: Action: <action> | Subjects: <subjects> | Mood: <mood>"
                    ),
                })

                if not isinstance(self.client, VLLMClient):
                    raise RuntimeError("VideoVLM requires a VLLMClient instance")
                    
                return await self.client._chat_completion([{"role": "user", "content": content}])

            try:
                output_text = await _do_request(sampled)
            except Exception as exc:
                if "At most 1 image" in str(exc) and len(sampled) > 1:
                    log.warning("[VideoVLM] vLLM server limits to 1 image per prompt; falling back to single middle keyframe.")
                    middle_frame = [sampled[len(sampled) // 2]]
                    output_text = await _do_request(middle_frame)
                else:
                    raise

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
        pass
