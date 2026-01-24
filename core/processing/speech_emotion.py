"""Speech Emotion Recognition (SER) using Wav2Vec2.

Identifies emotional tone in voice segments (angry, happy, sad, neutral, etc.).
"""

from __future__ import annotations

import asyncio
from typing import Any
import numpy as np
from core.utils.logger import get_logger
import torch

log = get_logger(__name__)

class SpeechEmotionAnalyzer:
    """Analyzer for Speech Emotion Recognition."""

    def __init__(self, device: str | None = None):
        """Initialize SER analyzer."""
        self.model = None
        self.processor = None
        self._device = device
        self._init_lock = asyncio.Lock()
        
    def _get_device(self) -> str:
        if self._device:
            return self._device
        return "cuda" if torch.cuda.is_available() else "cpu"

    async def _lazy_load(self) -> bool:
        if self.model:
            return True
            
        async with self._init_lock:
            if self.model:
                return True
                
            try:
                from transformers import AutoModelForAudioClassification, Wav2Vec2FeatureExtractor
                
                # Using a popular fine-tuned model for emotion
                model_name = "ehcalabres/wav2vec2-lg-xlsr-en-speech-emotion-recognition"
                
                self.processor = Wav2Vec2FeatureExtractor.from_pretrained(model_name)
                self.model = AutoModelForAudioClassification.from_pretrained(model_name)
                
                device = self._get_device()
                self.model.to(device)
                self._device = device
                
                log.info(f"[SER] Model loaded on {device}")
                return True
                
            except Exception as e:
                log.error(f"[SER] Failed to load model: {e}")
                return False

    async def analyze(
        self, 
        audio_segment: np.ndarray, 
        sample_rate: int = 16000
    ) -> dict[str, Any]:
        """Analyze average emotion of an audio segment.
        
        Args:
            audio_segment: Audio data.
            sample_rate: Sampling rate (will be resampled to 16k).
        """
        if not await self._lazy_load():
            return {"emotion": "unknown", "confidence": 0.0}
            
        try:
            import librosa
            
            # Resample to 16kHz (Wav2Vec2 native)
            if sample_rate != 16000:
                audio_segment = librosa.resample(
                    audio_segment.astype(np.float32),
                    orig_sr=sample_rate, 
                    target_sr=16000
                )
                
            # Split into chunks if too long (>10s) to avoid OOM, or just take center crop
            # For efficiency and accuracy on "tone", center 5s is usually enough
            max_len = 16000 * 10
            if len(audio_segment) > max_len:
                center = len(audio_segment) // 2
                start = center - (max_len // 2)
                audio_segment = audio_segment[start : start + max_len]

            inputs = self.processor(
                audio_segment, 
                sampling_rate=16000, 
                return_tensors="pt", 
                padding=True
            )
            
            inputs = {k: v.to(self._device) for k, v in inputs.items()}
            
            with torch.no_grad():
                logits = self.model(**inputs).logits
                probs = torch.nn.functional.softmax(logits, dim=-1)
                
            # Get output labels from config
            id2label = self.model.config.id2label
            score, idx = torch.max(probs, dim=-1)
            
            emotion = id2label[idx.item()]
            confidence = score.item()
            
            return {
                "emotion": emotion,
                "confidence": round(confidence, 3),
                "scores": {id2label[i]: round(float(probs[0][i]), 3) for i in range(len(id2label))}
            }
            
        except Exception as e:
            log.warning(f"[SER] Analysis failed: {e}")
            return {"emotion": "unknown", "confidence": 0.0}
            
    def cleanup(self):
        if self.model:
            del self.model
        if self.processor:
            del self.processor
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
