"""Video-Native VLM Client (Qwen2-VL / LLaVA-Video).

Provides true video understanding by processing multiple frames with temporal queries.
Backs "Action Summary" feature to fix "posing to camera" hallucinations.
"""

from __future__ import annotations

import asyncio

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoProcessor

from core.utils.logger import get_logger
from core.utils.resource_arbiter import GPU_SEMAPHORE

log = get_logger(__name__)


class VideoVLM:
    """Video Understanding VLM (Qwen2-VL)."""

    def __init__(self, model_id: str | None = None):
        from config import settings
        self.model_id = model_id or settings.video_vlm_model_id
        self.model = None
        self.processor = None
        self._init_lock = asyncio.Lock()
        self._device = "cuda" if torch.cuda.is_available() else "cpu"

    async def _lazy_load(self) -> bool:
        """Load model with Flash Attention and Quantization."""
        if self.model is not None:
            return True

        async with self._init_lock:
            if self.model is not None:
                return True
            try:
                from core.utils.resource_arbiter import RESOURCE_ARBITER
                
                from config import settings
                vram_gb = getattr(settings, 'video_vlm_vram_gb', 4.0)
                
                if not await RESOURCE_ARBITER.ensure_loaded(
                    "video_vlm", 
                    vram_gb=vram_gb, 
                    cleanup_fn=self.cleanup
                ):
                     log.error("[VideoVLM] Failed to acquire VRAM")
                     return False

                log.info(f"[VideoVLM] Loading {self.model_id}...")

                # Check for Flash Attention 2
                attn_impl = (
                    "flash_attention_2"
                    if torch.cuda.get_device_capability()[0] >= 8
                    else "eager"
                )

                self.model = AutoModelForCausalLM.from_pretrained(
                    self.model_id,
                    torch_dtype=torch.bfloat16
                    if torch.cuda.is_bf16_supported()
                    else torch.float16,
                    attn_implementation=attn_impl,
                    device_map="auto",
                    trust_remote_code=True,
                )

                self.processor = AutoProcessor.from_pretrained(
                    self.model_id, trust_remote_code=True
                )

                log.info(
                    f"[VideoVLM] Loaded on {self._device} with {attn_impl}"
                )
                return True
            except Exception as e:
                log.error(f"[VideoVLM] Load failed: {e}")
                return False

    async def generate_action_summary(
        self, frames: list[np.ndarray], fps: float = 1.0
    ) -> dict[str, str]:
        """Generate structured action summary from video frames.

        Args:
            frames: List of RGB numpy frames.
            fps: Approximate frame rate of the list (e.g. 1.0 means 1 frame per sec).

        Returns:
            JSON-like dict with keys: 'action', 'subject', 'mood'.
        """
        if not await self._lazy_load():
            return {}

        async with GPU_SEMAPHORE:
            try:
                # Sampling: Qwen2-VL handles variable frames, but let's cap at 16 for memory
                from config import settings
                max_frames = settings.vlm_max_frames
                sampled_frames = frames
                if len(frames) > max_frames:
                    indices = np.linspace(0, len(frames) - 1, max_frames, dtype=int)
                    sampled_frames = [frames[i] for i in indices]

                # Prepare inputs (Qwen2-VL specific format)
                # Note: Actual implementation depends on qwen_vl_utils which is standard companion
                # But here we assume standard transformers processor usage if updated

                # Convert frames to standard format (local path or PIL)
                # Qwen2-VL can accept PIL images
                from PIL import Image

                pil_frames = [Image.fromarray(f) for f in sampled_frames]

                messages = [
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "video",
                                "video": pil_frames,
                                "fps": fps,  # Placeholder, processor handles it
                            },
                            {
                                "type": "text",
                                "text": (
                                    "Analyze this video clip. Describe the main action, "
                                    "the subjects involved, and the interaction. "
                                    "Do not describe valid static poses. Focus on movement."
                                    "Format: Action: ... | Subjects: ... | Mood: ..."
                                ),
                            },
                        ],
                    }
                ]

                text = self.processor.apply_chat_template(
                    messages, tokenize=False, add_generation_prompt=True
                )

                inputs = self.processor(
                    text=[text],
                    videos=[pil_frames],
                    padding=True,
                    return_tensors="pt",
                )
                inputs = inputs.to(self.model.device)

                generated_ids = self.model.generate(
                    **inputs, max_new_tokens=settings.vlm_max_tokens
                )
                generated_ids_trimmed = [
                    out_ids[len(in_ids) :]
                    for in_ids, out_ids in zip(
                        inputs.input_ids, generated_ids, strict=True
                    )
                ]
                output_text = self.processor.batch_decode(
                    generated_ids_trimmed,
                    skip_special_tokens=True,
                    clean_up_tokenization_spaces=False,
                )[0]

                # Simple parsing
                parts = output_text.split("|")
                result = {"raw": output_text}
                for part in parts:
                    if ":" in part:
                        k, v = part.split(":", 1)
                        result[k.strip().lower()] = v.strip()

                return result

            except Exception as e:
                log.error(f"[VideoVLM] Generation failed: {e}")
                return {}

    def cleanup(self):
        if self.model:
            del self.model
            self.model = None
        if self.processor:
            del self.processor
            self.processor = None
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
