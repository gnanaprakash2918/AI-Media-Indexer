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
        self._load_failed = False  # Prevents retry spam on permanent failures

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

        # Don't retry if we already failed (prevents spam)
        if self._load_failed:
            return False

        async with self._init_lock:
            if self.model is not None:
                return True

            if self._load_failed:
                return False

            try:
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
                self._load_failed = True
                return False
            except Exception as e:
                log.error(f"[CLAP] Failed to load model: {e}")
                self._load_failed = True
                return False

    async def detect_events(
        self,
        audio_segment: np.ndarray,
        target_classes: list[str],
        sample_rate: int = 16000,
        top_k: int = 3,
        threshold: float = 0.3,
        return_embedding: bool = False,
    ) -> list[dict] | tuple[list[dict], list[float] | None]:
        """Detect all audible events in the given segment.

        Args:
            audio_segment: The raw audio data.
            target_classes: A list of target event classes (descriptions) to detect.
            sample_rate: The sampling rate of the audio.
            top_k: The number of top events to return.
            threshold: The confidence threshold for an event to be considered detected.
            return_embedding: If True, also returns the 512-dim CLAP audio embedding.

        Returns:
            List of detected events with their labels and confidence scores.
            If return_embedding=True, returns (events, embedding) tuple.
        """
        if not target_classes:
            log.warning(
                "[CLAP] No target classes provided - open-vocabulary requires query-defined labels"
            )
            return ([], None) if return_embedding else []

        log.debug("[CLAP] detect_events: calling _lazy_load")
        if not await self._lazy_load():
            log.debug("[CLAP] detect_events: _lazy_load returned False")
            return ([], None) if return_embedding else []
        log.debug("[CLAP] detect_events: _lazy_load completed successfully")

        if self.model is None or self.processor is None:
            return ([], None) if return_embedding else []

        if audio_segment.size == 0:
            return ([], None) if return_embedding else []

        try:
            import torch

            from core.utils.resource_arbiter import RESOURCE_ARBITER

            # CLAP expects 48kHz audio - resample BEFORE acquiring GPU lock
            # to avoid blocking the async event loop
            target_sr = 48000
            if sample_rate != target_sr:
                log.debug(
                    f"[CLAP] Resampling audio from {sample_rate}Hz to {target_sr}Hz..."
                )
                import librosa

                audio_segment = librosa.resample(
                    audio_segment.astype(np.float32),
                    orig_sr=sample_rate,
                    target_sr=target_sr,
                )
                sample_rate = target_sr
                log.debug("[CLAP] Resampling complete")

            log.debug("[CLAP] Acquiring GPU for CLAP detection...")
            embedding_list = None
            async with RESOURCE_ARBITER.acquire("clap", vram_gb=1.0):
                log.debug("[CLAP] GPU acquired, processing...")
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

                log.debug("[CLAP] Running inference...")
                with torch.no_grad():
                    text_embeds = self.model.get_text_features(**text_inputs)
                    audio_embeds = self.model.get_audio_features(**audio_inputs)

                    # Store raw embedding before normalization if requested
                    if return_embedding:
                        embedding_list = audio_embeds.squeeze(0).cpu().numpy().tolist()

                    text_embeds = text_embeds / text_embeds.norm(
                        dim=-1, keepdim=True
                    )
                    audio_embeds = audio_embeds / audio_embeds.norm(
                        dim=-1, keepdim=True
                    )

                    similarity = (audio_embeds @ text_embeds.T).squeeze(0)
                    probs = similarity.softmax(dim=-1).cpu().numpy()
                log.debug("[CLAP] Inference complete")

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

            if return_embedding:
                return results, embedding_list
            return results

        except Exception as e:
            log.error(f"[CLAP] Detection failed: {e}")
            return ([], None) if return_embedding else []

    async def detect_events_batch(
        self,
        audio_chunks: list[tuple[np.ndarray, float]],
        target_classes: list[str],
        sample_rate: int = 16000,
        top_k: int = 3,
        threshold: float = 0.3,
        return_embedding: bool = False,
    ) -> list[list[dict]] | list[tuple[list[dict], list[float] | None]]:
        """Detect audio events in multiple chunks with a single GPU acquisition.

        This is significantly more efficient than calling detect_events() in a loop,
        as it resamples all audio upfront and processes all chunks in one GPU session.

        Args:
            audio_chunks: List of (audio_segment, start_time) tuples.
            target_classes: A list of target event classes to detect.
            sample_rate: The sampling rate of the audio.
            top_k: The number of top events to return per chunk.
            threshold: The confidence threshold for detection.
            return_embedding: If True, returns (events, embedding) tuple for each chunk.

        Returns:
            List of detected events per chunk, each with labels and confidence scores.
            If return_embedding=True, returns list of (events, embedding) tuples.
        """
        if not target_classes:
            log.warning("[CLAP] No target classes provided")
            if return_embedding:
                return [([], None) for _ in audio_chunks]
            return [[] for _ in audio_chunks]

        if not audio_chunks:
            return []

        if not await self._lazy_load():
            if return_embedding:
                return [([], None) for _ in audio_chunks]
            return [[] for _ in audio_chunks]

        if self.model is None or self.processor is None:
            if return_embedding:
                return [([], None) for _ in audio_chunks]
            return [[] for _ in audio_chunks]

        try:
            import torch

            from core.utils.resource_arbiter import RESOURCE_ARBITER

            # CLAP expects 48kHz audio - resample ALL chunks BEFORE acquiring GPU
            target_sr = 48000
            resampled_chunks = []
            log.info(
                f"[CLAP] Batch: Resampling {len(audio_chunks)} chunks from {sample_rate}Hz to {target_sr}Hz..."
            )

            import librosa

            for audio_segment, start_time in audio_chunks:
                if audio_segment.size == 0:
                    resampled_chunks.append((np.array([]), start_time))
                    continue

                if sample_rate != target_sr:
                    resampled = librosa.resample(
                        audio_segment.astype(np.float32),
                        orig_sr=sample_rate,
                        target_sr=target_sr,
                    )
                else:
                    resampled = audio_segment.astype(np.float32)
                resampled_chunks.append((resampled, start_time))

            log.info("[CLAP] Batch: Resampling complete")

            # Single GPU acquisition for ALL chunks
            log.info(
                f"[CLAP] Batch: Acquiring GPU for {len(resampled_chunks)} chunks..."
            )
            async with RESOURCE_ARBITER.acquire("clap", vram_gb=1.0):
                log.info("[CLAP] Batch: GPU acquired, processing all chunks...")
                device = self._device or "cpu"

                # Pre-compute text embeddings ONCE for all chunks
                text_inputs = self.processor(  # type: ignore
                    text=target_classes,
                    return_tensors="pt",
                    padding=True,
                )
                text_inputs = {k: v.to(device) for k, v in text_inputs.items()}

                with torch.no_grad():
                    text_embeds = self.model.get_text_features(**text_inputs)
                    text_embeds = text_embeds / text_embeds.norm(
                        dim=-1, keepdim=True
                    )

                all_results: list = []

                for audio_segment, _start_time in resampled_chunks:
                    if audio_segment.size == 0:
                        if return_embedding:
                            all_results.append(([], None))
                        else:
                            all_results.append([])
                        continue

                    audio_inputs = self.processor(  # type: ignore
                        audios=audio_segment,
                        sampling_rate=target_sr,
                        return_tensors="pt",
                    )
                    audio_inputs = {
                        k: v.to(device) for k, v in audio_inputs.items()
                    }

                    with torch.no_grad():
                        audio_embeds = self.model.get_audio_features(
                            **audio_inputs
                        )
                        
                        # Store embedding before normalization if requested
                        embedding_list = None
                        if return_embedding:
                            embedding_list = audio_embeds.squeeze(0).cpu().numpy().tolist()
                        
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
                    
                    if return_embedding:
                        all_results.append((results, embedding_list))
                    else:
                        all_results.append(results)

                log.info(f"[CLAP] Batch: Processed {len(all_results)} chunks")
                return all_results

        except Exception as e:
            log.error(f"[CLAP] Batch detection failed: {e}")
            return [[] for _ in audio_chunks]

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

    async def encode_text(self, text: str) -> list[float] | None:
        """Encode text to CLAP text embedding for semantic audio search.
        
        This produces a 512-dim text embedding that can be compared against
        CLAP audio embeddings stored in the audio_events collection.
        
        Args:
            text: Query text to encode.
            
        Returns:
            512-dim CLAP text embedding, or None on failure.
        """
        if not text:
            return None
            
        if not await self._lazy_load():
            return None
            
        if self.model is None or self.processor is None:
            return None
            
        try:
            import torch
            from core.utils.resource_arbiter import RESOURCE_ARBITER
            
            async with RESOURCE_ARBITER.acquire("clap", vram_gb=0.5):
                device = self._device or "cpu"
                
                text_inputs = self.processor(
                    text=[text],
                    return_tensors="pt",
                    padding=True,
                )
                text_inputs = {k: v.to(device) for k, v in text_inputs.items()}
                
                with torch.no_grad():
                    text_embeds = self.model.get_text_features(**text_inputs)
                    # Normalize for cosine similarity
                    text_embeds = text_embeds / text_embeds.norm(dim=-1, keepdim=True)
                    
                return text_embeds.squeeze(0).cpu().numpy().tolist()
                
        except Exception as e:
            log.error(f"[CLAP] Text encoding failed: {e}")
            return None

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
