"""Module for detecting audio events and matching them to descriptions."""
from __future__ import annotations

import asyncio
from typing import Any

import numpy as np

from core.utils.logger import get_logger

log = get_logger(__name__)


class AudioEventDetector:
    """Detector for identifying specific audio events (e.g., siren, applause)."""
    def __init__(self, device: str | None = None):
        """Initialize audio event detector."""
        self.model = None
        self.processor = None
        self._device: str | None = device
        self._init_lock = asyncio.Lock()

    def _get_device(self) -> str:
        if self._device:
            return self._device
        try:
            import torch

            return "cuda" if torch.cuda.is_available() else "cpu"
        except ImportError:
            return "cpu"

    async def _lazy_load(self) -> bool:
        if self.model is not None:
            return True

        async with self._init_lock:
            if self.model is not None:
                return True

            try:
                from core.utils.resource_arbiter import RESOURCE_ARBITER

                async with RESOURCE_ARBITER.acquire("clap", vram_gb=1.0):
                    log.info("[CLAP] Loading model...")

                    from transformers import ClapModel, ClapProcessor

                    self.processor = ClapProcessor.from_pretrained(
                        "laion/clap-htsat-unfused"
                    )
                    self.model = ClapModel.from_pretrained(
                        "laion/clap-htsat-unfused"
                    )

                    device = self._get_device()
                    self.model.to(device)  # type: ignore
                    self._device = device

                    log.info(f"[CLAP] Model loaded on {device}")
                    return True

            except ImportError as e:
                log.warning(f"[CLAP] transformers not available: {e}")
                return False
            except Exception as e:
                log.error(f"[CLAP] Failed to load model: {e}")
                return False

    async def detect_events(
        self,
        audio_segment: np.ndarray,
        target_classes: list[str],
        sample_rate: int = 16000,
        top_k: int = 3,
        threshold: float = 0.3,
    ) -> list[dict]:
        """Detect all audible events in the given segment.

        Args:
            audio_segment: The raw audio data.
            target_classes: A list of target event classes (descriptions) to detect.
            sample_rate: The sampling rate of the audio.
            top_k: The number of top events to return.
            threshold: The confidence threshold for an event to be considered detected.

        Returns:
            List of detected events with their labels and confidence scores.
        """
        if not target_classes:
            log.warning(
                "[CLAP] No target classes provided - open-vocabulary requires query-defined labels"
            )
            return []

        if not await self._lazy_load():
            return []

        if self.model is None or self.processor is None:
            return []

        if audio_segment.size == 0:
            return []

        try:
            import torch

            from core.utils.resource_arbiter import RESOURCE_ARBITER

            async with RESOURCE_ARBITER.acquire("clap", vram_gb=1.0):
                device = self._device or "cpu"

                text_inputs = self.processor(  # type: ignore
                    text=target_classes,
                    return_tensors="pt",
                    padding=True,
                )
                text_inputs = {k: v.to(device) for k, v in text_inputs.items()}

                audio_inputs = self.processor(  # type: ignore
                    audios=audio_segment,
                    sampling_rate=sample_rate,
                    return_tensors="pt",
                )
                audio_inputs = {
                    k: v.to(device) for k, v in audio_inputs.items()
                }

                with torch.no_grad():
                    text_embeds = self.model.get_text_features(**text_inputs)
                    audio_embeds = self.model.get_audio_features(**audio_inputs)

                    text_embeds = text_embeds / text_embeds.norm(
                        dim=-1, keepdim=True
                    )
                    audio_embeds = audio_embeds / audio_embeds.norm(
                        dim=-1, keepdim=True
                    )

                    similarity = (audio_embeds @ text_embeds.T).squeeze(0)
                    probs = similarity.softmax(dim=-1).cpu().numpy()

            results = []
            top_indices = probs.argsort()[::-1][:top_k]
            for idx in top_indices:
                conf = float(probs[idx])
                if conf >= threshold:
                    results.append(
                        {
                            "event": target_classes[idx],
                            "confidence": round(conf, 3),
                        }
                    )

            if results:
                log.debug(f"[CLAP] Detected: {results}")

            return results

        except Exception as e:
            log.error(f"[CLAP] Detection failed: {e}")
            return []

    async def match_audio_description(
        self,
        audio_segment: np.ndarray,
        target_description: str,
        sample_rate: int = 16000,
    ) -> dict[str, Any]:
        """Check if the audio segment matches a semantic description.

        Args:
            audio_segment: Raw audio data.
            target_description: Natural language description of the sound to match.
            sample_rate: The sampling rate of the audio.

        Returns:
            Match result with similarity score and a boolean indicating a match.
        """
        if not target_description:
            return {"score": 0.0, "error": "No description provided"}

        results = await self.detect_events(
            audio_segment,
            target_classes=[target_description],
            sample_rate=sample_rate,
            top_k=1,
            threshold=0.0,
        )

        if results:
            return {
                "target": target_description,
                "score": results[0]["confidence"],
                "match": results[0]["confidence"] > 0.3,
            }
        return {"target": target_description, "score": 0.0, "match": False}

    async def compare_audio_candidates(
        self,
        audio_segment: np.ndarray,
        candidate_descriptions: list[str],
        sample_rate: int = 16000,
    ) -> list[dict[str, Any]]:
        """Compare audio segment against candidate descriptions using CLAP."""
        if not candidate_descriptions:
            return []

        results = await self.detect_events(
            audio_segment,
            target_classes=candidate_descriptions,
            sample_rate=sample_rate,
            top_k=len(candidate_descriptions),
            threshold=0.0,
        )

        sorted_results = []
        for desc in candidate_descriptions:
            found = next((r for r in results if r["event"] == desc), None)
            if found:
                sorted_results.append(
                    {"label": desc, "score": found["confidence"]}
                )
            else:
                sorted_results.append({"label": desc, "score": 0.0})

        sorted_results.sort(key=lambda x: x["score"], reverse=True)
        return sorted_results

    def should_run_clap(self, vad_result: dict) -> bool:
        """Check if clap detection should run based on VAD."""
        is_speech = vad_result.get("is_speech", False)
        speech_conf = vad_result.get("speech_confidence", 0)
        if is_speech and speech_conf > 0.9:
            return False
        return True

    def cleanup(self) -> None:
        """Release model resources."""
        if self.model is not None:
            del self.model
            self.model = None
        if self.processor is not None:
            del self.processor
            self.processor = None

        try:
            import torch

            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except ImportError:
            pass

        log.info("[CLAP] Resources released")
