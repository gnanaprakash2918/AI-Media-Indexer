"""Audio loudness/decibel analysis for hyper-granular search.

Measures audio levels in dB/LUFS to enable queries like:
- "crowd cheer peaking at 92dB"
- "whispered conversation"
- "loud explosion"

Uses ITU-R BS.1770-4 standard via pyloudnorm.
"""

from __future__ import annotations

import asyncio
from typing import Any

import numpy as np

from core.utils.logger import get_logger

log = get_logger(__name__)


class AudioLoudnessAnalyzer:
    """Analyze audio loudness for hyper-granular search.
    
    Usage:
        analyzer = AudioLoudnessAnalyzer()
        result = await analyzer.analyze(audio_segment, sample_rate=16000)
        # {"lufs": -14.2, "peak_db": -3.1, "estimated_spl": 85}
    """
    
    def __init__(self, sample_rate: int = 16000):
        """Initialize loudness analyzer.
        
        Args:
            sample_rate: Expected audio sample rate.
        """
        self.sample_rate = sample_rate
        self._meter = None
        self._init_lock = asyncio.Lock()
    
    async def _lazy_load(self) -> bool:
        """Lazy load pyloudnorm meter."""
        if self._meter is not None:
            return True
        
        async with self._init_lock:
            if self._meter is not None:
                return True
            
            try:
                import pyloudnorm as pyln
                
                self._meter = pyln.Meter(self.sample_rate)
                log.info(f"[Loudness] Meter initialized (SR={self.sample_rate})")
                return True
                
            except ImportError:
                log.warning("[Loudness] pyloudnorm not installed")
                return False
            except Exception as e:
                log.error(f"[Loudness] Failed to initialize: {e}")
                return False
    
    async def analyze(
        self,
        audio: np.ndarray,
        sample_rate: int | None = None,
    ) -> dict[str, Any]:
        """Analyze audio segment loudness.
        
        Args:
            audio: Audio waveform as float32 numpy array.
            sample_rate: Optional sample rate override.
        
        Returns:
            Dict with:
                - lufs: Integrated loudness (LUFS)
                - peak_db: True peak in dB
                - rms_db: RMS level in dB
                - dynamic_range: Loudness Range (LRA)
                - estimated_spl: Rough environmental SPL estimate
                - loudness_category: "quiet", "moderate", "loud", "very_loud"
        """
        if not await self._lazy_load():
            return self._empty_result()
        
        sr = sample_rate or self.sample_rate
        
        # Resample if needed
        if sr != self.sample_rate:
            try:
                import librosa
                audio = librosa.resample(audio, orig_sr=sr, target_sr=self.sample_rate)
            except ImportError:
                log.warning("[Loudness] librosa not available for resampling")
        
        try:
            # Ensure float32 and correct shape
            audio = np.asarray(audio, dtype=np.float32)
            if len(audio.shape) > 1:
                audio = audio.mean(axis=-1)  # Mono
            
            # Compute integrated loudness (LUFS)
            lufs = self._meter.integrated_loudness(audio)
            
            # True peak
            peak = 20 * np.log10(np.max(np.abs(audio)) + 1e-10)
            
            # RMS level
            rms = np.sqrt(np.mean(audio ** 2))
            rms_db = 20 * np.log10(rms + 1e-10)
            
            # Dynamic range (simple approximation)
            # Compute short-term loudness variance
            window_size = int(0.4 * self.sample_rate)  # 400ms windows
            dynamic_range = self._compute_dynamic_range(audio, window_size)
            
            # Estimate environmental SPL (rough approximation)
            # LUFS to SPL mapping based on typical calibration
            estimated_spl = self._estimate_spl(lufs)
            
            # Categorize loudness
            category = self._categorize_loudness(estimated_spl)
            
            return {
                "lufs": float(lufs) if not np.isnan(lufs) else -70.0,
                "peak_db": float(peak),
                "rms_db": float(rms_db),
                "dynamic_range": float(dynamic_range),
                "estimated_spl": estimated_spl,
                "loudness_category": category,
            }
            
        except Exception as e:
            log.error(f"[Loudness] Analysis failed: {e}")
            return self._empty_result()
    
    def _compute_dynamic_range(
        self,
        audio: np.ndarray,
        window_size: int,
    ) -> float:
        """Compute dynamic range using windowed RMS."""
        # Compute RMS for each window
        n_windows = len(audio) // window_size
        if n_windows < 2:
            return 0.0
        
        rms_values = []
        for i in range(n_windows):
            window = audio[i * window_size:(i + 1) * window_size]
            rms = np.sqrt(np.mean(window ** 2))
            rms_db = 20 * np.log10(rms + 1e-10)
            rms_values.append(rms_db)
        
        # Dynamic range = difference between 95th and 10th percentile
        return float(np.percentile(rms_values, 95) - np.percentile(rms_values, 10))
    
    def _estimate_spl(self, lufs: float) -> int:
        """Estimate SPL from LUFS (rough calibration).
        
        This is a very rough estimate. Actual SPL depends on
        playback equipment and environment.
        
        Common reference: -23 LUFS ≈ 83 dB SPL (broadcast standard)
        """
        if np.isnan(lufs) or lufs < -60:
            return 30  # Near silence
        
        # Linear mapping from LUFS to approximate SPL
        # -60 LUFS → ~30 dB SPL (quiet room)
        # -23 LUFS → ~83 dB SPL (reference)
        # -10 LUFS → ~96 dB SPL (loud)
        # 0 LUFS → ~106 dB SPL (very loud)
        
        spl = 83 + (lufs + 23) * 1.5  # Rough linear mapping
        return max(20, min(120, int(spl)))  # Clamp to reasonable range
    
    def _categorize_loudness(self, spl: int) -> str:
        """Categorize SPL into human-readable levels."""
        if spl < 40:
            return "very_quiet"
        elif spl < 60:
            return "quiet"
        elif spl < 75:
            return "moderate"
        elif spl < 90:
            return "loud"
        else:
            return "very_loud"
    
    def _empty_result(self) -> dict[str, Any]:
        """Return empty result on failure."""
        return {
            "lufs": -70.0,
            "peak_db": -70.0,
            "rms_db": -70.0,
            "dynamic_range": 0.0,
            "estimated_spl": 30,
            "loudness_category": "unknown",
        }
    
    async def detect_loud_moments(
        self,
        audio: np.ndarray,
        threshold_spl: int = 85,
        min_duration_ms: int = 200,
    ) -> list[dict[str, Any]]:
        """Find moments where audio exceeds threshold.
        
        Useful for queries like "crowd cheer at 92dB".
        
        Args:
            audio: Audio waveform.
            threshold_spl: SPL threshold to detect.
            min_duration_ms: Minimum duration to count as event.
        
        Returns:
            List of {start, end, peak_spl, category} dicts.
        """
        if not await self._lazy_load():
            return []
        
        # Convert SPL threshold to approximate LUFS
        target_lufs = (threshold_spl - 83) / 1.5 - 23
        
        # Analyze in 100ms windows
        window_ms = 100
        window_samples = int(window_ms / 1000 * self.sample_rate)
        min_windows = max(1, min_duration_ms // window_ms)
        
        loud_moments = []
        current_loud_start = None
        current_peak = -70
        consecutive_loud = 0
        
        for i in range(0, len(audio), window_samples):
            window = audio[i:i + window_samples]
            if len(window) < window_samples // 2:
                continue
            
            # Quick RMS check (faster than full loudness)
            rms = np.sqrt(np.mean(window ** 2))
            rms_db = 20 * np.log10(rms + 1e-10)
            
            # Approximate LUFS from RMS (rough)
            approx_lufs = rms_db - 10  # Very rough approximation
            
            if approx_lufs > target_lufs:
                if current_loud_start is None:
                    current_loud_start = i / self.sample_rate
                consecutive_loud += 1
                current_peak = max(current_peak, approx_lufs)
            else:
                if consecutive_loud >= min_windows:
                    end_time = i / self.sample_rate
                    peak_spl = self._estimate_spl(current_peak)
                    loud_moments.append({
                        "start": current_loud_start,
                        "end": end_time,
                        "peak_spl": peak_spl,
                        "category": self._categorize_loudness(peak_spl),
                    })
                
                current_loud_start = None
                current_peak = -70
                consecutive_loud = 0
        
        # Handle last segment
        if consecutive_loud >= min_windows and current_loud_start is not None:
            end_time = len(audio) / self.sample_rate
            peak_spl = self._estimate_spl(current_peak)
            loud_moments.append({
                "start": current_loud_start,
                "end": end_time,
                "peak_spl": peak_spl,
                "category": self._categorize_loudness(peak_spl),
            })
        
        log.info(f"[Loudness] Found {len(loud_moments)} moments above {threshold_spl} dB SPL")
        return loud_moments
