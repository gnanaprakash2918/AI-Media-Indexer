"""Core system components for media indexing and retrieval."""

import os
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

# FIX: torchaudio >= 2.6 compatibility patch for pyannote.audio
from dataclasses import dataclass
import torchaudio

if not hasattr(torchaudio, "AudioMetaData"):
    @dataclass
    class AudioMetaData:
        sample_rate: int
        num_frames: int
        num_channels: int
        bits_per_sample: int = 16
        encoding: str = "PCM_S"
    torchaudio.AudioMetaData = AudioMetaData

if not hasattr(torchaudio, "list_audio_backends"):
    torchaudio.list_audio_backends = lambda: ["soundfile"]

if not hasattr(torchaudio, "backend"):
    class _BackendCommon:
        AudioMetaData = AudioMetaData
    class _Backend:
        common = _BackendCommon
    torchaudio.backend = _Backend

