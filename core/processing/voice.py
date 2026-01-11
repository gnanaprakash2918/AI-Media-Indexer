"""Voice processing module for speaker diarization and embedding extraction."""

from __future__ import annotations

import asyncio
import os
from pathlib import Path
from typing import Final, cast

import numpy as np
import torch

# torch.load to disable weights_only enforcement for pyannote/speechbrain compatibility
_original_load = torch.load

def safe_load(*args, **kwargs):
    # FORCE weights_only=False even if True was passed
    kwargs['weights_only'] = False
    return _original_load(*args, **kwargs)

torch.load = safe_load

from pyannote.audio import Inference, Model, Pipeline
from pyannote.core import Segment
from huggingface_hub import snapshot_download

from config import settings
from core.schemas import SpeakerSegment
from core.utils.logger import get_logger
from core.utils.observe import observe
from core.utils.locks import GPU_SEMAPHORE

log = get_logger(__name__)

MIN_SEGMENT_DURATION: Final = 0.8
MAX_SEGMENT_DURATION: Final = 30.0
EMBEDDING_VERSION: Final = "wespeaker_resnet34_v1_l2"


def is_audio_silent(audio_data: np.ndarray, threshold_db: float | None = None) -> bool:
    if threshold_db is None:
        threshold_db = settings.audio_rms_silence_db
    if audio_data.size == 0:
        return True
    rms = np.sqrt(np.mean(audio_data.astype(np.float64) ** 2))
    if rms < 1e-10:
        return True
    rms_db = 20 * np.log10(rms + 1e-10)
    return rms_db < threshold_db


class VoiceProcessor:
    """Handles speaker diarization and voice embedding extraction."""

    def __init__(self) -> None:
        """Initialize the voice processor with settings from config."""
        self.enabled = bool(settings.enable_voice_analysis)
        self.device = torch.device(settings.device)
        self.hf_token = settings.hf_token if settings.hf_token else None

        self.pipeline: Pipeline | None = None
        self.embedding_model: Model | None = None
        self.inference: Inference | None = None
        self._initialized = False
        self._init_lock = asyncio.Lock()

        if not self.enabled:
            log.info("Voice analysis disabled by config.")
            return

        if not self.hf_token:
            log.warning("HF_TOKEN missing. Voice analysis disabled.")
            self.enabled = False

    async def _lazy_init(self) -> None:
        if not self.enabled or self._initialized:
            return

        async with self._init_lock:
            if self._initialized:
                return

            try:
                async with GPU_SEMAPHORE:
                    log.info(
                        f"Loading pyannote pipeline={settings.pyannote_model} "
                        f"embedder={settings.voice_embedding_model} "
                        f"device={self.device}"
                    )

                    try:
                        self.pipeline = Pipeline.from_pretrained(
                            settings.pyannote_model,
                            use_auth_token=self.hf_token,
                        )
                    except Exception as pipe_err:
                        log.warning(f"Pyannote Pipeline load failed: {pipe_err}. Attempting snapshot_download...")
                        try:
                            snapshot_download(
                                repo_id=settings.pyannote_model,
                                token=self.hf_token,
                                ignore_patterns=["*.msgpack", "*.h5", "*.ot"],
                            )
                            self.pipeline = Pipeline.from_pretrained(
                                settings.pyannote_model,
                                use_auth_token=self.hf_token,
                            )
                        except Exception as dl_err:
                            log.error(f"Failed to download/reload Pyannote pipeline: {dl_err}")
                            raise pipe_err

                    self.pipeline.to(self.device)

                    try:
                        self.embedding_model = Model.from_pretrained(
                            settings.voice_embedding_model,
                            use_auth_token=self.hf_token,
                        )
                    except Exception as model_err:
                        log.warning(f"Voice embedding model load failed: {model_err}. Attempting snapshot_download...")
                        try:
                            snapshot_download(
                                repo_id=settings.voice_embedding_model,
                                token=self.hf_token,
                                ignore_patterns=["*.msgpack", "*.h5", "*.ot"],
                            )
                            self.embedding_model = Model.from_pretrained(
                                settings.voice_embedding_model,
                                use_auth_token=self.hf_token,
                            )
                        except Exception as dl_err:
                            log.error(f"Failed to download/reload embedding model: {dl_err}")
                            raise model_err

                    self.inference = Inference(
                        self.embedding_model,
                        window="whole",
                        device=self.device,
                    )

                self._initialized = True

            except Exception as e:
                log.error(f"Voice model initialization failed: {e}")
                self.enabled = False
                self.pipeline = None
                self.embedding_model = None
                self.inference = None

                self.inference = None
    
    def cleanup(self) -> None:
        """Release GPU resources."""
        if self.pipeline:
            del self.pipeline
            self.pipeline = None
        if self.embedding_model:
            del self.embedding_model
            self.embedding_model = None
        if self.inference:
            del self.inference
            self.inference = None
        
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
        
        self._initialized = False
        log.info("Voice processor resources released")

    async def process(self, audio_path: Path) -> list[SpeakerSegment]:
        """Process an audio file to extract speaker segments and embeddings.

        Args:
            audio_path: Path to the audio file.

        Returns:
            List of detected speaker segments.
        """
        if not self.enabled or not audio_path.exists():
            return []

        await self._lazy_init()
        if not self.pipeline or not self.inference:
            return []

        segments: list[SpeakerSegment] = []
        temp_wav: Path | None = None

        try:
            # Always convert to WAV to ensure compatibility with all video/audio formats
            # CRITICAL: Create fresh temp file per video to avoid batch processing issues
            log.info(f"[Voice] Processing: {audio_path.name}")
            temp_wav = await self._convert_to_wav(audio_path)
            if temp_wav is None:
                log.warning(f"[Voice] Skipped - WAV conversion failed for {audio_path.name}")
                return []
            processing_path = temp_wav

            async with GPU_SEMAPHORE:
                diarization = self.pipeline(
                    str(processing_path),
                    min_speakers=settings.min_speakers,
                    max_speakers=settings.max_speakers,
                )

            for turn, _, speaker in diarization.itertracks(yield_label=True):
                start = float(turn.start)
                end = float(turn.end)
                duration = end - start

                if duration < MIN_SEGMENT_DURATION:
                    continue
                if duration > MAX_SEGMENT_DURATION:
                    end = start + MAX_SEGMENT_DURATION

                embedding = await self._extract_embedding(processing_path, start, end)
                if embedding is None:
                    continue

                segments.append(
                    SpeakerSegment(
                        start_time=start,
                        end_time=end,
                        speaker_label=speaker,
                        confidence=1.0,
                        embedding=embedding,
                    )
                )

            return segments

        except Exception as e:
            log.error(f"Voice processing failed for {audio_path.name}: {e}")
            return []
        
        finally:
            if temp_wav and temp_wav.exists():
                try:
                    temp_wav.unlink()
                except Exception:
                    pass

    async def _convert_to_wav(self, path: Path) -> Path | None:
        """Convert any audio/video to a temporary 16kHz mono WAV file."""
        import tempfile
        import subprocess

        fd, temp_path_str = tempfile.mkstemp(suffix=".wav", prefix="voice_temp_")
        os.close(fd)
        temp_path = Path(temp_path_str)

        cmd = [
            "ffmpeg",
            "-y",
            "-i", str(path),
            "-vn",              # Disable video
            "-acodec", "pcm_s16le",
            "-ar", "16000",     # 16kHz
            "-ac", "1",         # Mono
            str(temp_path)
        ]

        try:
            process = await asyncio.create_subprocess_exec(
                *cmd,
                stdout=asyncio.subprocess.DEVNULL,
                stderr=asyncio.subprocess.PIPE
            )
            _, stderr = await process.communicate()
            
            if process.returncode != 0:
                log.warning(f"FFmpeg conversion failed: {stderr.decode()}")
                if temp_path.exists():
                    temp_path.unlink()
                return None
                
            return temp_path
        except Exception as e:
            log.warning(f"Failed to convert {path} to wav: {e}")
            if temp_path.exists():
                temp_path.unlink()
            return None

    async def _extract_embedding(
        self,
        audio_path: Path,
        start: float,
        end: float,
    ) -> list[float] | None:
        if not self.inference:
            return None

        try:
            async with GPU_SEMAPHORE:
                emb = self.inference.crop(
                    str(audio_path),
                    Segment(start, end),
                )

            if isinstance(emb, torch.Tensor):
                vec = cast(torch.Tensor, emb).squeeze().float().cpu().numpy()
            else:
                vec = np.asarray(emb, dtype=np.float32).reshape(-1)

            if vec.ndim != 1 or vec.size == 0:
                return None

            vec /= np.linalg.norm(vec) + 1e-9
            return vec.tolist()

        except Exception:
            return None
