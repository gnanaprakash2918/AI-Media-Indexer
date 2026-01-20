"""Music structure analysis for temporal precision in searches.

Detects verse, chorus, bridge, drop, and other structural segments in music
to enable precise temporal queries like "during the chorus" or "at the drop".
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from core.utils.logger import get_logger

log = get_logger(__name__)


@dataclass
class MusicSection:
    """A detected section in a music track."""
    
    section_type: str  # verse, chorus, bridge, intro, outro, drop, breakdown
    start_time: float
    end_time: float
    confidence: float = 0.8
    beat_count: int = 0
    tempo: float = 0.0
    energy: float = 0.0  # Average energy level (0-1)
    
    @property
    def duration(self) -> float:
        return self.end_time - self.start_time
    
    def to_dict(self) -> dict[str, Any]:
        return {
            "section_type": self.section_type,
            "start_time": round(self.start_time, 2),
            "end_time": round(self.end_time, 2),
            "duration": round(self.duration, 2),
            "confidence": round(self.confidence, 3),
            "beat_count": self.beat_count,
            "tempo": round(self.tempo, 1),
            "energy": round(self.energy, 3),
        }


@dataclass
class MusicAnalysis:
    """Complete structural analysis of a music track."""
    
    sections: list[MusicSection] = field(default_factory=list)
    global_tempo: float = 0.0
    key: str = ""
    duration: float = 0.0
    has_vocals: bool = False
    energy_curve: list[float] = field(default_factory=list)
    beat_times: list[float] = field(default_factory=list)
    
    def get_section_at_time(self, timestamp: float) -> MusicSection | None:
        """Find the section containing the given timestamp."""
        for section in self.sections:
            if section.start_time <= timestamp <= section.end_time:
                return section
        return None
    
    def get_sections_by_type(self, section_type: str) -> list[MusicSection]:
        """Get all sections of a specific type (e.g., all choruses)."""
        return [s for s in self.sections if s.section_type.lower() == section_type.lower()]
    
    def to_dict(self) -> dict[str, Any]:
        return {
            "global_tempo": round(self.global_tempo, 1),
            "key": self.key,
            "duration": round(self.duration, 2),
            "has_vocals": self.has_vocals,
            "section_count": len(self.sections),
            "sections": [s.to_dict() for s in self.sections],
            "beat_times": [round(t, 3) for t in self.beat_times[:100]],  # Limit for storage
        }


class MusicStructureAnalyzer:
    """Analyzes music structure to detect verses, choruses, bridges, etc.
    
    Uses energy analysis, spectral contrast, and self-similarity matrices
    to segment music into structural sections.
    
    Usage:
        analyzer = MusicStructureAnalyzer()
        analysis = analyzer.analyze_file(Path("song.mp3"))
        
        # Find all choruses
        choruses = analysis.get_sections_by_type("chorus")
        
        # Get section at specific timestamp
        section = analysis.get_section_at_time(45.0)
    """
    
    def __init__(
        self,
        hop_length: int = 512,
        n_fft: int = 2048,
        segment_duration: float = 4.0,  # Minimum section length
    ):
        """Initialize the music structure analyzer.
        
        Args:
            hop_length: Hop length for spectral analysis.
            n_fft: FFT window size.
            segment_duration: Minimum duration for a detected section.
        """
        self.hop_length = hop_length
        self.n_fft = n_fft
        self.segment_duration = segment_duration
        self._initialized = False
    
    def _lazy_init(self) -> bool:
        """Check if librosa is available."""
        if self._initialized:
            return True
        
        try:
            import librosa
            self._initialized = True
            return True
        except ImportError:
            log.warning("[MusicStructure] librosa not available")
            return False
    
    def analyze_file(self, audio_path: Path) -> MusicAnalysis:
        """Analyze a music file and detect structural sections.
        
        Args:
            audio_path: Path to the audio file.
            
        Returns:
            MusicAnalysis object with detected sections.
        """
        if not self._lazy_init():
            return MusicAnalysis()
        
        if not audio_path.exists():
            log.warning(f"[MusicStructure] File not found: {audio_path}")
            return MusicAnalysis()
        
        try:
            import librosa
            
            log.info(f"[MusicStructure] Analyzing: {audio_path.name}")
            
            # Load audio
            y, sr = librosa.load(str(audio_path), sr=22050)
            duration = librosa.get_duration(y=y, sr=sr)
            
            # Get tempo and beat frames
            tempo, beat_frames = librosa.beat.beat_track(y=y, sr=sr, hop_length=self.hop_length)
            beat_times = librosa.frames_to_time(beat_frames, sr=sr, hop_length=self.hop_length).tolist()
            
            # Compute features for segmentation
            sections = self._detect_sections(y, sr, tempo, beat_times, duration)
            
            # Detect if vocals are present (using spectral centroid variance)
            has_vocals = self._detect_vocals(y, sr)
            
            # Compute energy curve for temporal queries
            energy_curve = self._compute_energy_curve(y, sr)
            
            analysis = MusicAnalysis(
                sections=sections,
                global_tempo=float(tempo) if isinstance(tempo, (int, float)) else float(tempo[0]) if hasattr(tempo, '__len__') else 120.0,
                duration=duration,
                has_vocals=has_vocals,
                energy_curve=energy_curve,
                beat_times=beat_times,
            )
            
            log.info(f"[MusicStructure] Found {len(sections)} sections at {analysis.global_tempo:.1f} BPM")
            return analysis
            
        except Exception as e:
            log.error(f"[MusicStructure] Analysis failed: {e}")
            return MusicAnalysis()
    
    def analyze_array(
        self, 
        audio: np.ndarray, 
        sr: int = 22050
    ) -> MusicAnalysis:
        """Analyze audio array directly.
        
        Args:
            audio: Audio samples as numpy array.
            sr: Sample rate.
            
        Returns:
            MusicAnalysis object with detected sections.
        """
        if not self._lazy_init():
            return MusicAnalysis()
        
        try:
            import librosa
            
            duration = len(audio) / sr
            
            # Get tempo and beat frames
            tempo, beat_frames = librosa.beat.beat_track(y=audio, sr=sr, hop_length=self.hop_length)
            beat_times = librosa.frames_to_time(beat_frames, sr=sr, hop_length=self.hop_length).tolist()
            
            sections = self._detect_sections(audio, sr, tempo, beat_times, duration)
            has_vocals = self._detect_vocals(audio, sr)
            energy_curve = self._compute_energy_curve(audio, sr)
            
            return MusicAnalysis(
                sections=sections,
                global_tempo=float(tempo) if isinstance(tempo, (int, float)) else float(tempo[0]) if hasattr(tempo, '__len__') else 120.0,
                duration=duration,
                has_vocals=has_vocals,
                energy_curve=energy_curve,
                beat_times=beat_times,
            )
            
        except Exception as e:
            log.error(f"[MusicStructure] Array analysis failed: {e}")
            return MusicAnalysis()
    
    def _detect_sections(
        self,
        y: np.ndarray,
        sr: int,
        tempo: float,
        beat_times: list[float],
        duration: float,
    ) -> list[MusicSection]:
        """Detect structural sections using energy and spectral features.
        
        Uses a combination of:
        1. Energy/RMS changes to detect transitions
        2. Spectral contrast for verse vs chorus differentiation
        3. Self-similarity matrix for repetition detection
        """
        import librosa
        
        sections: list[MusicSection] = []
        
        # Compute features
        rms = librosa.feature.rms(y=y, hop_length=self.hop_length)[0]
        spectral_contrast = librosa.feature.spectral_contrast(y=y, sr=sr, hop_length=self.hop_length)
        mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13, hop_length=self.hop_length)
        
        # Compute self-similarity matrix for structure detection
        chroma = librosa.feature.chroma_cqt(y=y, sr=sr, hop_length=self.hop_length)
        
        # Detect segment boundaries using spectral clustering
        try:
            # Use librosa's segment detection
            bounds = librosa.segment.agglomerative(chroma, k=None)
            bound_times = librosa.frames_to_time(bounds, sr=sr, hop_length=self.hop_length)
        except Exception:
            # Fallback: divide into fixed segments
            n_segments = max(4, int(duration / 30))  # ~30 second segments
            bound_times = np.linspace(0, duration, n_segments + 1)
        
        # Convert boundaries to sections
        for i in range(len(bound_times) - 1):
            start = float(bound_times[i])
            end = float(bound_times[i + 1])
            
            if end - start < self.segment_duration:
                continue
            
            # Classify section type based on features
            start_frame = librosa.time_to_frames(start, sr=sr, hop_length=self.hop_length)
            end_frame = librosa.time_to_frames(end, sr=sr, hop_length=self.hop_length)
            end_frame = min(end_frame, len(rms))
            
            if start_frame >= end_frame:
                continue
            
            # Calculate segment energy
            segment_energy = float(np.mean(rms[start_frame:end_frame]))
            avg_energy = float(np.mean(rms))
            
            # Calculate spectral brightness (high = chorus, low = verse)
            segment_contrast = float(np.mean(spectral_contrast[:, start_frame:end_frame]))
            
            # Classify based on energy and position
            section_type = self._classify_section(
                energy=segment_energy,
                avg_energy=avg_energy,
                contrast=segment_contrast,
                position_ratio=start / duration,
                is_first=(i == 0),
                is_last=(i == len(bound_times) - 2),
            )
            
            # Count beats in this section
            beat_count = sum(1 for t in beat_times if start <= t <= end)
            
            sections.append(MusicSection(
                section_type=section_type,
                start_time=start,
                end_time=end,
                confidence=0.75,  # Base confidence
                beat_count=beat_count,
                tempo=float(tempo) if isinstance(tempo, (int, float)) else 120.0,
                energy=segment_energy / max(avg_energy, 0.001),  # Normalized energy
            ))
        
        # Post-process: enhance confidence for repeated patterns
        sections = self._enhance_section_labels(sections)
        
        return sections
    
    def _classify_section(
        self,
        energy: float,
        avg_energy: float,
        contrast: float,
        position_ratio: float,
        is_first: bool,
        is_last: bool,
    ) -> str:
        """Classify a section based on its features and position."""
        
        # Position-based classification
        if is_first and position_ratio < 0.1:
            return "intro"
        if is_last and position_ratio > 0.85:
            return "outro"
        
        # Energy-based classification
        energy_ratio = energy / max(avg_energy, 0.001)
        
        if energy_ratio > 1.4:
            # High energy - likely chorus or drop
            if contrast > 0.5:
                return "drop"
            return "chorus"
        elif energy_ratio < 0.6:
            # Low energy - likely bridge or breakdown
            if position_ratio > 0.5:
                return "bridge"
            return "breakdown"
        elif energy_ratio < 0.85:
            # Medium-low energy - likely verse
            return "verse"
        else:
            # Medium-high energy - could be pre-chorus or chorus
            return "chorus" if contrast > 0.3 else "pre-chorus"
    
    def _enhance_section_labels(self, sections: list[MusicSection]) -> list[MusicSection]:
        """Enhance section labels based on patterns and repetition."""
        if len(sections) < 3:
            return sections
        
        # Look for verse-chorus patterns
        # Typically: intro, verse, chorus, verse, chorus, bridge, chorus, outro
        
        # Count occurrences to find the likely chorus
        type_counts = {}
        for s in sections:
            type_counts[s.section_type] = type_counts.get(s.section_type, 0) + 1
        
        # The most energetic repeated section is likely the chorus
        chorus_candidates = [s for s in sections if s.section_type == "chorus"]
        if len(chorus_candidates) >= 2:
            # Increase confidence for repeated choruses
            for s in chorus_candidates:
                s.confidence = min(0.9, s.confidence + 0.1)
        
        return sections
    
    def _detect_vocals(self, y: np.ndarray, sr: int) -> bool:
        """Detect if the audio contains vocals using spectral features."""
        import librosa
        
        try:
            # Vocals typically have high spectral centroid variance
            centroid = librosa.feature.spectral_centroid(y=y, sr=sr)[0]
            centroid_var = float(np.var(centroid))
            
            # Vocals also have specific harmonic content
            harmonic, _ = librosa.effects.hpss(y)
            harmonic_ratio = float(np.sum(np.abs(harmonic))) / (float(np.sum(np.abs(y))) + 1e-6)
            
            # Heuristic: high variance and harmonic content suggests vocals
            has_vocals = centroid_var > 1e6 and harmonic_ratio > 0.3
            return has_vocals
            
        except Exception:
            return False
    
    def _compute_energy_curve(
        self, 
        y: np.ndarray, 
        sr: int,
        n_points: int = 100,
    ) -> list[float]:
        """Compute a simplified energy curve for the entire track."""
        import librosa
        
        try:
            rms = librosa.feature.rms(y=y, hop_length=self.hop_length)[0]
            
            # Downsample to n_points
            if len(rms) > n_points:
                indices = np.linspace(0, len(rms) - 1, n_points, dtype=int)
                energy_curve = rms[indices].tolist()
            else:
                energy_curve = rms.tolist()
            
            # Normalize to 0-1
            max_energy = max(energy_curve) if energy_curve else 1.0
            if max_energy > 0:
                energy_curve = [e / max_energy for e in energy_curve]
            
            return energy_curve
            
        except Exception:
            return []
    
    def get_high_energy_moments(
        self,
        analysis: MusicAnalysis,
        threshold: float = 0.8,
    ) -> list[tuple[float, float]]:
        """Get timestamps of high-energy moments (drops, climaxes).
        
        Args:
            analysis: MusicAnalysis object.
            threshold: Energy threshold (0-1).
            
        Returns:
            List of (start_time, end_time) tuples for high-energy sections.
        """
        moments = []
        for section in analysis.sections:
            if section.energy >= threshold:
                moments.append((section.start_time, section.end_time))
        return moments
    
    def get_section_for_query(
        self,
        analysis: MusicAnalysis,
        query: str,
    ) -> list[MusicSection]:
        """Find sections matching a temporal query.
        
        Args:
            analysis: MusicAnalysis object.
            query: Natural language query like "during the chorus", "at the drop".
            
        Returns:
            List of matching sections.
        """
        query_lower = query.lower()
        
        # Map query terms to section types
        section_mappings = {
            "chorus": ["chorus"],
            "verse": ["verse"],
            "bridge": ["bridge"],
            "intro": ["intro"],
            "outro": ["outro"],
            "drop": ["drop", "breakdown"],
            "breakdown": ["breakdown"],
            "climax": ["chorus", "drop"],
            "beginning": ["intro"],
            "ending": ["outro"],
            "hook": ["chorus"],
            "refrain": ["chorus"],
        }
        
        for term, types in section_mappings.items():
            if term in query_lower:
                matching = []
                for t in types:
                    matching.extend(analysis.get_sections_by_type(t))
                if matching:
                    return matching
        
        return []


# Global instance for convenience
_analyzer: MusicStructureAnalyzer | None = None


def get_music_analyzer() -> MusicStructureAnalyzer:
    """Get or create the global music structure analyzer."""
    global _analyzer
    if _analyzer is None:
        _analyzer = MusicStructureAnalyzer()
    return _analyzer
