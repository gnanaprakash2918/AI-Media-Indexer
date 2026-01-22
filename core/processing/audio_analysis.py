"""Audio Analysis: Tempo, Beat, Mood Detection (TikTok style).

Enables audio-driven search like "high energy music" or "slow sad song".

Based on Research:
- TikTok Search: Audio Beat/Tempo Sync
- Netflix Internal: Mood classification
"""

from __future__ import annotations

import asyncio
from typing import Any

import numpy as np

from core.utils.logger import get_logger

log = get_logger(__name__)


class AudioTempoAnalyzer:
    """Analyze audio tempo, beat, and mood.

    Uses librosa for beat detection and tempo estimation.
    Enables queries like "upbeat music" or "slow tempo scenes".

    Usage:
        analyzer = AudioTempoAnalyzer()
        result = await analyzer.analyze(audio_segment, sample_rate)
        # {"tempo": 120.0, "beat_strength": 0.8, "mood": "energetic"}
    """

    def __init__(self):
        """Initialize audio tempo analyzer."""
        self._librosa = None
        self._init_lock = asyncio.Lock()

    async def _lazy_load(self) -> bool:
        """Load librosa lazily."""
        if self._librosa is not None:
            return True

        async with self._init_lock:
            if self._librosa is not None:
                return True

            try:
                import librosa

                self._librosa = librosa
                log.info("[AudioTempo] librosa loaded")
                return True
            except ImportError:
                log.warning("[AudioTempo] librosa not installed")
                return False

    async def _classify_mood_sota(
        self,
        audio_segment: np.ndarray,
        sample_rate: int,
    ) -> tuple[str, float]:
        """Classify mood using SOTA Zero-Shot Audio Classification (CLAP).

        Uses LAION-CLAP or similar models to match audio against
        text descriptions of moods in a zero-shot manner.

        Args:
            audio_segment: Audio samples.
            sample_rate: Sample rate.

        Returns:
            Tuple of (mood_label, confidence_score).
        """
        try:
            # Lazy load CLAP via our centralized AudioEventDetector to save VRAM
            # (Reusing existing component instead of loading new model)
            from core.processing.audio_events import AudioEventDetector

            # We use a singleton or new instance depending on how it's managed
            # Here we instantiate but it lazy loads internally
            detector = AudioEventDetector()

            # Define mood prompts for zero-shot classification
            mood_prompts = [
                "energetic upbeat music",
                "sad melancholic music",
                "tense dramatic suspenseful audio",
                "calm peaceful relaxing music",
                "romantic emotional music",
                "aggressive intense angry audio",
                "neutral background noise",
                "happy cheerful music",
            ]

            # Use the detector's zero-shot capability
            # Note: We need to ensure text_queries are supported in classify_segment
            # If not, we might need to extend AudioEventDetector or use it directly here

            # For now, let's implement a direct zero-shot flow assuming we can access the model
            # This follows the pattern in VideoUnderstanding where we reuse encoders

            if not await detector._lazy_load():
                log.warning(
                    "[AudioTempo] CLAP model not available for SOTA mood"
                )
                return "unknown", 0.0

            import torch
            import librosa

            # Resample for CLAP (48kHz)
            if sample_rate != 48000:
                audio_segment = librosa.resample(
                    audio_segment.astype(np.float32),
                    orig_sr=sample_rate,
                    target_sr=48000,
                )

            # Prepare inputs
            audio_inputs = detector.processor(
                audios=audio_segment,
                sampling_rate=48000,
                return_tensors="pt",
                padding=True,
            )
            text_inputs = detector.processor(
                text=mood_prompts, return_tensors="pt", padding=True
            )

            audio_inputs = {
                k: v.to(detector._device) for k, v in audio_inputs.items()
            }
            text_inputs = {
                k: v.to(detector._device) for k, v in text_inputs.items()
            }

            with torch.no_grad():
                # Get embeddings
                audio_embed = detector.model.get_audio_features(**audio_inputs)
                text_embed = detector.model.get_text_features(**text_inputs)

                # Normalize
                audio_embed = audio_embed / audio_embed.norm(
                    dim=-1, keepdim=True
                )
                text_embed = text_embed / text_embed.norm(dim=-1, keepdim=True)

                # Cosine similarity
                similarity = (audio_embed @ text_embed.T).squeeze()

                # Get best match
                score, idx = similarity.max(dim=0)
                best_mood = mood_prompts[idx.item()]
                confidence = float(score.item())

                # Clean up prompt to get label
                # "energetic upbeat music" -> "energetic"
                simple_mood = best_mood.split()[0]

                log.info(
                    f"[AudioTempo] SOTA Mood: '{simple_mood}' "
                    f"(Original: '{best_mood}', Conf: {confidence:.2f})"
                )

                return simple_mood, confidence

        except Exception as e:
            log.error(f"[AudioTempo] SOTA Mood classification failed: {e}")
            return "unknown", 0.0

    async def analyze(
        self,
        audio_segment: np.ndarray,
        sample_rate: int = 22050,
    ) -> dict[str, Any]:
        """Analyze audio using SOTA methods.

        Args:
            audio_segment: Audio samples.
            sample_rate: Sample rate.

        Returns:
            Dict with tempo, beat info, and SOTA mood.
        """
        log.debug(
            f"[AudioTempo] Analyzing segment: shape={audio_segment.shape}, sr={sample_rate}"
        )

        if not await self._lazy_load():
            return {"tempo": 0, "error": "librosa/deps not available"}

        try:
            librosa = self._librosa

            # Ensure mono
            if len(audio_segment.shape) > 1:
                audio = audio_segment.mean(axis=1)
            else:
                audio = audio_segment

            # 1. Feature Extraction (Traditional but fast/robust for tempo)
            tempo, beat_frames = librosa.beat.beat_track(
                y=audio, sr=sample_rate
            )
            onset_env = librosa.onset.onset_strength(y=audio, sr=sample_rate)
            beat_strength = (
                float(onset_env.mean()) if len(onset_env) > 0 else 0.0
            )
            rms = librosa.feature.rms(y=audio)
            energy = float(rms.mean()) if rms.size > 0 else 0

            # 2. SOTA Mood Classification (Zero-Shot CLAP)
            # This replaces the rule-based _classify_mood
            mood, mood_conf = await self._classify_mood_sota(audio, sample_rate)

            # Fallback to rule-based if SOTA fails or confidence is low?
            # Actually, let's trust the SOTA model or fallback to 'neutral'
            if mood == "unknown":
                # Fallback to simple features if CLAP fails
                spectral_centroid = librosa.feature.spectral_centroid(
                    y=audio, sr=sample_rate
                )
                avg_centroid = (
                    float(spectral_centroid.mean())
                    if spectral_centroid.size > 0
                    else 0
                )
                mood = self._classify_mood_rule_based(
                    float(tempo), energy, avg_centroid
                )
                log.info(f"[AudioTempo] Fallback to rule-based mood: {mood}")

            tempo_val = round(float(tempo), 1)
            tempo_cat = self._tempo_category(tempo_val)

            log.info(
                f"[AudioTempo] Analysis Complete: "
                f"Tempo={tempo_val}bpm ({tempo_cat}), "
                f"Mood={mood} (conf={mood_conf:.2f}), "
                f"Energy={energy:.3f}"
            )

            return {
                "tempo": tempo_val,
                "tempo_category": tempo_cat,
                "beat_strength": round(beat_strength, 3),
                "beat_count": len(beat_frames),
                "energy": round(energy, 4),
                "mood": mood,
                "mood_confidence": round(mood_conf, 3),
            }

        except Exception as e:
            log.exception(f"[AudioTempo] Critical analysis failure: {e}")
            return {"tempo": 0, "error": str(e)}

    def _classify_mood_rule_based(
        self,
        tempo: float,
        energy: float,
        spectral_centroid: float,
    ) -> str:
        """Legacy rule-based fallback."""
        is_fast = tempo > 120
        is_high_energy = energy > 0.1
        is_bright = spectral_centroid > 2000

        if is_fast and is_high_energy:
            return "energetic"
        elif is_fast and not is_high_energy:
            return "upbeat"
        elif not is_fast and is_high_energy:
            return "intense"
        elif not is_fast and not is_high_energy and is_bright:
            return "calm"
        elif not is_fast and not is_high_energy and not is_bright:
            return "melancholic"
        else:
            return "neutral"

    async def detect_music_vs_speech(
        self,
        audio_segment: np.ndarray,
        sample_rate: int = 22050,
    ) -> dict[str, float]:
        """Detect whether audio is primarily music or speech.

        Args:
            audio_segment: Audio samples.
            sample_rate: Sample rate.

        Returns:
            Dict with music_probability and speech_probability.
        """
        if not await self._lazy_load():
            return {"music": 0.5, "speech": 0.5}

        try:
            librosa = self._librosa

            if len(audio_segment.shape) > 1:
                audio = audio_segment.mean(axis=1)
            else:
                audio = audio_segment

            # Music tends to have more consistent spectral features
            # Speech has more variation

            # Zero crossing rate (higher for speech)
            zcr = librosa.feature.zero_crossing_rate(audio)
            avg_zcr = float(zcr.mean())

            # Spectral flatness (higher for noise/speech harmonics)
            flatness = librosa.feature.spectral_flatness(y=audio)
            avg_flatness = float(flatness.mean())

            # Simple heuristic
            # Music: lower ZCR, lower flatness
            # Speech: higher ZCR, higher flatness

            music_score = 1.0 - (avg_zcr * 10 + avg_flatness * 2)
            music_score = max(0, min(1, music_score))

            return {
                "music": round(music_score, 3),
                "speech": round(1 - music_score, 3),
            }

        except Exception as e:
            log.error(f"[AudioTempo] Music/Speech detection failed: {e}")
            return {"music": 0.5, "speech": 0.5}


class SaliencyDetector:
    """Visual saliency detection (Bing Video Search style).

    Identifies the most "important" regions of a frame
    for thumbnail selection and attention-based search.
    """

    def __init__(self):
        """Initialize saliency detector."""
        self._cv2 = None

    def _ensure_cv2(self) -> bool:
        """Ensure OpenCV is available."""
        if self._cv2 is not None:
            return True

        try:
            import cv2

            self._cv2 = cv2
            return True
        except ImportError:
            log.warning("[Saliency] OpenCV not available")
            return False

    async def compute_saliency(
        self,
        frame: np.ndarray,
    ) -> np.ndarray | None:
        """Compute saliency map for a frame.

        Args:
            frame: RGB frame as numpy array.

        Returns:
            Saliency map (same size as frame, grayscale).
        """
        if not self._ensure_cv2():
            return None

        try:
            cv2 = self._cv2

            # Use spectral residual saliency
            saliency = cv2.saliency.StaticSaliencySpectralResidual_create()

            # Compute
            success, saliency_map = saliency.computeSaliency(frame)

            if success:
                # Normalize to 0-255
                saliency_map = (saliency_map * 255).astype(np.uint8)
                return saliency_map

            return None

        except Exception as e:
            log.error(f"[Saliency] Computation failed: {e}")
            return None

    async def get_attention_region(
        self,
        frame: np.ndarray,
    ) -> dict[str, Any]:
        """Get the most salient region of the frame.

        Args:
            frame: RGB frame.

        Returns:
            Dict with bbox of attention region and score.
        """
        saliency_map = await self.compute_saliency(frame)

        if saliency_map is None:
            return {"bbox": None, "score": 0}

        try:
            cv2 = self._cv2

            # Threshold to find salient regions
            _, thresh = cv2.threshold(saliency_map, 127, 255, cv2.THRESH_BINARY)

            # Find contours
            contours, _ = cv2.findContours(
                thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
            )

            if not contours:
                return {"bbox": None, "score": 0}

            # Get largest contour
            largest = max(contours, key=cv2.contourArea)
            x, y, w, h = cv2.boundingRect(largest)

            # Calculate average saliency in this region
            region = saliency_map[y : y + h, x : x + w]
            score = float(region.mean()) / 255.0

            return {
                "bbox": [x, y, x + w, y + h],
                "score": round(score, 3),
            }

        except Exception as e:
            log.error(f"[Saliency] Region detection failed: {e}")
            return {"bbox": None, "score": 0}

    async def select_best_thumbnail(
        self,
        frames: list[np.ndarray],
    ) -> int:
        """Select the best frame for a thumbnail based on saliency.

        Args:
            frames: List of candidate frames.

        Returns:
            Index of the best frame.
        """
        if not frames:
            return 0

        best_idx = 0
        best_score = 0.0

        for i, frame in enumerate(frames):
            result = await self.get_attention_region(frame)
            score = result.get("score", 0)

            if score > best_score:
                best_score = score
                best_idx = i

        return best_idx
