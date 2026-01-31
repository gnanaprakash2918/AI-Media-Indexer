"""Voice processing module for speaker diarization and embedding extraction."""

from __future__ import annotations

import asyncio
import os
from pathlib import Path
from typing import Any, Final, cast

import numpy as np
import torch

# torch.load to disable weights_only enforcement for pyannote/speechbrain compatibility
_original_load = torch.load


def safe_load(*args: Any, **kwargs: Any) -> Any:
    """Wraps torch.load to force weights_only=False for older models.

    This is necessary for compatibility with pyannote and speechbrain
    models which may still use pickle-based serialization.
    """
    # FORCE weights_only=False even if True was passed
    if "weights_only" in kwargs:
        kwargs["weights_only"] = False
    return _original_load(*args, **kwargs)


torch.load = safe_load

from huggingface_hub import snapshot_download  # noqa: E402
from pyannote.audio import Inference, Model, Pipeline  # noqa: E402
from pyannote.core import Segment  # noqa: E402

from config import settings  # noqa: E402
from core.schemas import SpeakerSegment  # noqa: E402
from core.utils.logger import get_logger  # noqa: E402
from core.utils.resource_arbiter import GPU_SEMAPHORE  # noqa: E402

log = get_logger(__name__)

MIN_SEGMENT_DURATION: Final = 0.8
MAX_SEGMENT_DURATION: Final = 30.0
MIN_EMBEDDING_DURATION: Final = 1.5
EMBEDDING_VERSION: Final = "wespeaker_resnet34_v1_l2"


def is_audio_silent(
    audio_data: np.ndarray, threshold_db: float | None = None
) -> bool:
    """Detects if an audio segment is silent based on an RMS threshold.

    Args:
        audio_data: Numpy array of audio samples.
        threshold_db: Optional silence threshold in decibels.
            If None, uses settings.audio_rms_silence_db.

    Returns:
        True if the audio power is below the threshold.
    """
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

    def __init__(self, db: Any = None) -> None:
        """Initializes the voice processor.

        Configuration is read from the global settings. Analysis is only
        enabled if both enable_voice_analysis is True and a HuggingFace
        token is provided.

        Args:
            db: Optional vector database interface for storing embeddings.
        """
        self.db = db
        self.enabled = bool(settings.enable_voice_analysis)
        self.device = torch.device(settings.device)
        self.hf_token = settings.hf_token if settings.hf_token else None

        self.pipeline: Pipeline | None = None
        self.embedding_model: Model | None = None
        self.inference: Inference | None = None
        self._initialized = False
        self._init_lock = asyncio.Lock()

        # Audio cache for load-once optimization (avoids repeated I/O)
        self._cached_audio: tuple[str, np.ndarray, int] | None = (
            None  # (path, data, sample_rate)
        )

        if not self.enabled:
            log.info("Voice analysis disabled by config.")
            return

            log.warning("HF_TOKEN missing. Voice analysis disabled.")
            self.enabled = False

    def _has_audio_stream(self, input_path: Path) -> bool:
        """Checks if the file contains an audio stream using ffprobe."""
        try:
            import subprocess

            cmd = [
                "ffprobe",
                "-v",
                "error",
                "-select_streams",
                "a",
                "-show_entries",
                "stream=codec_type",
                "-of",
                "csv=p=0",
                str(input_path),
            ]
            output = (
                subprocess.check_output(cmd, stderr=subprocess.DEVNULL)
                .decode()
                .strip()
            )
            return bool(output)
        except Exception:
            return False

    def _get_cached_audio(self, path: Path) -> tuple[np.ndarray, int] | None:
        """Get audio from cache, loading if necessary.

        Optimization: Loads audio file once into memory, reuses for all segment slicing.

        Args:
            path: Path to audio file.

        Returns:
            Tuple of (audio_data, sample_rate) or None on failure.
        """
        path_str = str(path)

        # Return cached if same file
        if self._cached_audio and self._cached_audio[0] == path_str:
            return (self._cached_audio[1], self._cached_audio[2])

        try:
            import soundfile as sf

            audio_data, sample_rate = sf.read(path_str, dtype="float32")

            # Convert stereo to mono if needed
            if len(audio_data.shape) > 1:
                audio_data = np.mean(audio_data, axis=1)

            self._cached_audio = (path_str, audio_data, sample_rate)
            log.debug(
                f"[Voice] Cached audio: {path.name} ({len(audio_data) / sample_rate:.1f}s)"
            )
            return (audio_data, sample_rate)
        except Exception as e:
            log.warning(f"[Voice] Failed to cache audio {path.name}: {e}")
            return None

    def _slice_audio_memory(
        self, audio_data: np.ndarray, sample_rate: int, start: float, end: float
    ) -> np.ndarray:
        """Slice audio segment from in-memory buffer (fast, no I/O).

        Args:
            audio_data: Full audio array.
            sample_rate: Sample rate of audio.
            start: Start time in seconds.
            end: End time in seconds.

        Returns:
            Sliced audio segment as numpy array.
        """
        start_sample = int(start * sample_rate)
        end_sample = int(end * sample_rate)

        # Clamp to valid range
        start_sample = max(0, start_sample)
        end_sample = min(len(audio_data), end_sample)

        return audio_data[start_sample:end_sample]

    def _clear_audio_cache(self) -> None:
        """Clear the audio cache to free memory."""
        self._cached_audio = None

        if not self.enabled:
            log.info("Voice analysis disabled by config.")
            return

        if not self.hf_token:
            log.warning("HF_TOKEN missing. Voice analysis disabled.")
            self.enabled = False

    async def _lazy_init(self) -> None:
        """Loads heavy pyannote models only safe-guarded by a lock.

        Deals with model downloading and GPU allocation via the GPU_SEMAPHORE.
        """
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
                        log.warning(
                            f"Pyannote Pipeline load failed: {pipe_err}. Attempting snapshot_download..."
                        )
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
                            log.error(
                                f"Failed to download/reload Pyannote pipeline: {dl_err}"
                            )
                            raise pipe_err from dl_err

                    self.pipeline.to(self.device)

                    try:
                        self.embedding_model = Model.from_pretrained(
                            settings.voice_embedding_model,
                            use_auth_token=self.hf_token,
                        )
                    except Exception as model_err:
                        log.warning(
                            f"Voice embedding model load failed: {model_err}. Attempting snapshot_download..."
                        )
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
                            log.error(
                                f"Failed to download/reload embedding model: {dl_err}"
                            )
                            raise model_err from dl_err

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
        """Releases all voice models and clears GPU memory.

        Moves models to CPU, deletes references, and clears the torch CUDA cache.
        """
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
        """Performs speaker diarization and embedding extraction on an audio file."""
        if not self.enabled or not audio_path.exists():
            if not self.enabled:
                log.warning("[Voice] SKIPPED - Voice analysis disabled")
            else:
                log.warning(f"[Voice] SKIPPED - File not found: {audio_path}")
            return []

        await self._lazy_init()
        if not self.pipeline or not self.inference:
            log.warning(
                "[Voice] SKIPPED - Pipeline or inference model failed to load"
            )
            return []

        segments: list[SpeakerSegment] = []
        temp_wav: Path | None = None

        try:
            # Always convert to WAV to ensure compatibility with all video/audio formats
            # CRITICAL: Check for audio stream first to avoid FFmpeg crashes
            if not self._has_audio_stream(audio_path):
                log.warning(
                    f"[Voice] SKIPPED - No audio stream detected in {audio_path.name}"
                )
                return []

            log.info(f"[Voice] Processing: {audio_path.name}")
            temp_wav = await self._convert_to_wav(audio_path)
            if temp_wav is None:
                log.warning(
                    f"[Voice] Skipped - WAV conversion failed for {audio_path.name}"
                )
                return []
            processing_path = temp_wav

            async with GPU_SEMAPHORE:
                diarization = self.pipeline(
                    str(processing_path),
                    min_speakers=settings.min_speakers,
                    max_speakers=settings.max_speakers,
                )

            track_count = 0
            segments_with_placeholder = 0
            for turn, _, speaker in diarization.itertracks(yield_label=True):
                track_count += 1
                start = float(turn.start)
                end = float(turn.end)
                duration = end - start

                if duration < MIN_SEGMENT_DURATION:
                    # log.warning(f"Skipping short segment: {duration:.2f}s")
                    continue
                if duration > MAX_SEGMENT_DURATION:
                    end = start + MAX_SEGMENT_DURATION

                # Check for silence/garbage audio using cached audio (OPTIMIZED)
                # Pyannote is good but can hallucinate speech on noise or breaths
                is_silence_segment = False
                try:
                    # Use cached audio (loads once, slices in-memory for all segments)
                    cached = self._get_cached_audio(processing_path)
                    if cached:
                        audio_data, sr = cached
                        audio_chunk = self._slice_audio_memory(
                            audio_data, sr, start, end
                        )
                        if is_audio_silent(audio_chunk):
                            is_silence_segment = True
                    else:
                        # Fallback: read from file if caching failed
                        import soundfile as sf

                        audio_chunk, _ = sf.read(
                            str(processing_path),
                            start=int(start * 16000),
                            frames=int((end - start) * 16000),
                        )
                        if is_audio_silent(audio_chunk):
                            is_silence_segment = True
                except Exception as e:
                    log.warning(f"[Voice] Silence check failed: {e}")

                embedding = None
                if not is_silence_segment:
                    embedding = await self._extract_embedding(
                        processing_path, start, end
                    )
                else:
                    speaker = "SILENCE"  # Hardcoded category as requested
                    # Embedding remains None

                # If embedding extraction fails (and not silent), use a placeholder
                # This ensures segments are still stored for music/singing
                if embedding is None and not is_silence_segment:
                    log.warning(
                        f"[Voice] Embedding extraction failed for {start:.2f}-{end:.2f}s, using placeholder"
                    )
                    # Create a zeroed placeholder embedding (256-dim for wespeaker)
                    embedding = [0.0] * 256
                    segments_with_placeholder += 1

                segments.append(
                    SpeakerSegment(
                        start_time=start,
                        end_time=end,
                        speaker_label=speaker,
                        confidence=1.0
                        if is_silence_segment
                        else (1.0 if segments_with_placeholder == 0 else 0.5),
                        embedding=embedding,
                    )
                )

            log.info(
                f"[Voice] Found {len(segments)} segments out of {track_count} tracks "
                + f"({segments_with_placeholder} with placeholder embeddings)"
            )

            return segments

        except Exception as e:
            log.error(f"Voice processing failed for {audio_path.name}: {e}")
            return []

        finally:
            # Clear audio cache to free memory
            self._clear_audio_cache()

            if temp_wav and temp_wav.exists():
                try:
                    temp_wav.unlink()
                except Exception:
                    pass

    async def _convert_to_wav(self, path: Path) -> Path | None:
        """Converts an input media file to a standardized 16kHz mono WAV file.

        Args:
            path: Input file path (any media supported by ffmpeg).

        Returns:
            Path to the temporary WAV file, or None if conversion fails.
        """
        import tempfile

        fd, temp_path_str = tempfile.mkstemp(
            suffix=".wav", prefix="voice_temp_"
        )
        os.close(fd)
        temp_path = Path(temp_path_str)

        cmd = [
            "ffmpeg",
            "-y",
            "-i",
            str(path),
            "-vn",  # Disable video
            "-acodec",
            "pcm_s16le",
            "-ar",
            "16000",  # 16kHz
            "-ac",
            "1",  # Mono
            str(temp_path),
        ]

        try:
            process = await asyncio.create_subprocess_exec(
                *cmd,
                stdout=asyncio.subprocess.DEVNULL,
                stderr=asyncio.subprocess.PIPE,
            )
            _, stderr = await process.communicate()

            if process.returncode != 0:
                log.warning(
                    f"FFmpeg conversion failed: {stderr.decode(errors='replace')}"
                )
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
        """Extracts a voice embedding for a specific segment of audio.

        Args:
            audio_path: Path to the standardized WAV file.
            start: Start time in seconds.
            end: End time in seconds.

        Returns:
            A list of float values representing the embedding, or None if extraction fails.
        """
        if not self.inference:
            log.warning(
                "[Voice] Embedding extraction failed: inference model not initialized"
            )
            return None

        try:
            # Get actual audio duration to prevent boundary errors
            import soundfile as sf

            try:
                info = sf.info(str(audio_path))
                max_duration = float(info.duration)
            except Exception as sf_err:
                log.warning(
                    f"[Voice] Could not determine audio duration: {sf_err}, using segment end as fallback"
                )
                max_duration = end + 1.0

            # Clamp boundaries with safety margin (50ms) to prevent frame overflow
            safety_margin = 0.05
            original_start, original_end = start, end

            # Pad short segments to improve embedding quality
            duration = end - start
            if duration < MIN_EMBEDDING_DURATION:
                padding_needed = MIN_EMBEDDING_DURATION - duration
                half_pad = padding_needed / 2.0

                # Center pad within available audio
                start = max(0.0, start - half_pad)
                end = min(max_duration - safety_margin, end + half_pad)

                # If still too short (e.g. at start/end of file), extend the other side
                if (end - start) < MIN_EMBEDDING_DURATION:
                    if start == 0.0:
                        end = min(
                            max_duration - safety_margin,
                            start + MIN_EMBEDDING_DURATION,
                        )
                    elif end >= max_duration - safety_margin:
                        start = max(0.0, end - MIN_EMBEDDING_DURATION)
            else:
                # Just clamp standard margin
                start = max(0.0, start)
                end = min(max_duration - safety_margin, end)

            # Ensure minimum segment duration (100ms) -- fallback
            if end - start < 0.1:
                log.warning(
                    f"[Voice] Segment too short after clamping: "
                    f"original=[{original_start:.2f}, {original_end:.2f}], "
                    f"clamped=[{start:.2f}, {end:.2f}], max_duration={max_duration:.2f}"
                )
                return None

            async with GPU_SEMAPHORE:
                emb = self.inference.crop(
                    str(audio_path),
                    Segment(start, end),
                )

            if isinstance(emb, torch.Tensor):
                vec = cast(torch.Tensor, emb).squeeze().float().cpu().numpy()
            else:
                # Direct numpy array support from some pipelines
                vec = np.asarray(emb, dtype=np.float32).reshape(-1)

            if vec.ndim != 1 or vec.size == 0:
                log.warning(
                    f"[Voice] Invalid embedding shape: ndim={vec.ndim}, size={vec.size}"
                )
                return None

            vec /= np.linalg.norm(vec) + 1e-9
            return vec.tolist()

        except Exception as e:
            log.warning(
                f"[Voice] Embedding extraction error for {start:.2f}-{end:.2f}s: {e}"
            )
            return None
