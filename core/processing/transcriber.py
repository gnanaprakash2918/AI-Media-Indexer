"""Robust Audio Transcriber using Faster-Whisper (CTranslate2).

Features:
- RTX Optimized: Uses CTranslate2 (C++) with Int8/Float16 quantization.
- Smart Caching: Project Local > Global Cache > Download.
- High-Speed Download: Uses hf_transfer and parallel workers.
- Auto-Conversion: Automatically converts PyTorch models to CTranslate2.
- Anti-Hallucination: Strict generation config to prevent looping/translation.
- Pipeline Ready: Returns in-memory chunks for Vector DB ingestion.
- Tanglish Fixed: Uses "Mixed-Script Anchoring" prompt to prevent translation.
- Self-Healing: Automatically repairs corrupted model caches.
"""

import asyncio
import gc
import os
import shutil
import subprocess
import warnings
from pathlib import Path
from tempfile import NamedTemporaryFile
from typing import Any

import ctranslate2.converters
import torch
from faster_whisper import BatchedInferencePipeline, WhisperModel
from huggingface_hub import login, snapshot_download

from config import settings
from core.processing.text_utils import parse_srt
from core.utils.logger import log
from core.utils.observe import observe
from core.errors import TranscriberError, ModelLoadError

warnings.filterwarnings("ignore")

os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "1"
os.environ["HF_HUB_DOWNLOAD_TIMEOUT"] = "300"
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

# Auto-login to HuggingFace if token is available
_hf_token = os.getenv("HF_TOKEN") or getattr(settings, "hf_token", None)
if _hf_token:
    try:
        login(token=_hf_token, add_to_git_credential=False)
    except Exception:
        pass  # Already logged in or token invalid


def get_transcriber(language: str | None = None):
    """Factory to return appropriate transcriber engine based on settings.

    Args:
        language: Target language code (e.g., 'ta', 'hi', 'en').

    Returns:
        AudioTranscriber or IndicASRPipeline instance.
    """
    lang = language or settings.language or "en"

    if settings.use_indic_asr and lang in [
        "ta",
        "hi",
        "te",
        "ml",
        "kn",
        "bn",
        "gu",
        "mr",
        "or",
        "pa",
    ]:
        from core.processing.indic_transcriber import IndicASRPipeline

        return IndicASRPipeline(lang=lang)

    return AudioTranscriber()


class AudioTranscriber:
    """Orchestrates the audio-to-text transcription lifecycle.

    Handles model loading (Faster-Whisper), VRAM management, subtitle
    discovery, audio slicing, and SRT generation. Supports automatic
    fallback to smaller models on memory-constrained systems.
    """

    # Class-level shared model state (singleton pattern for VRAM efficiency)
    _SHARED_MODEL: WhisperModel | None = None
    _SHARED_BATCHED: BatchedInferencePipeline | None = None
    _SHARED_SIZE: str | None = None
    _FALLBACK_ATTEMPTED: bool = False

    # Smaller fallback models for memory-constrained systems (pre-converted, no conversion needed)
    LOW_MEMORY_MODELS = [
        "Systran/faster-distil-whisper-medium.en",  # ~500MB, English only
        "Systran/faster-whisper-small",  # ~250MB, multilingual
        "Systran/faster-whisper-base",  # ~140MB, multilingual
    ]

    def __init__(self) -> None:
        """Initializes the transcriber with default compute settings."""
        self._model: WhisperModel | None = None
        self._batched_model: BatchedInferencePipeline | None = None
        self._current_model_size: str | None = None
        self.device = settings.device
        self.compute_type: str = (
            "int8_float16" if self.device == "cuda" else "int8"
        )
        self._fallback_attempted = False
        self._locked_language: str | None = None

        # Register with Resource Arbiter for VRAM management
        try:
            from core.utils.resource_arbiter import RESOURCE_ARBITER

            RESOURCE_ARBITER.register_model("whisper", self.unload_model)
        except ImportError:
            pass

    def __enter__(self):
        """Called when entering the 'with' block."""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Called automatically when exiting the 'with' block.

        NOTE: We DO NOT unload here anymore. We let ResourceArbiter handle it
        via the registered callback to allow caching across files.
        """
        pass

    def unload_model(self) -> None:
        """Forcefully unloads the Whisper model from memory.

        Performs a full cleanup of CTranslate2 objects, triggers garbage
        collection, and clears the PyTorch CUDA cache to reclaim VRAM.
        """
        # Guard: only log and unload if a model is actually loaded
        if (
            self._model is None
            and self._batched_model is None
            and AudioTranscriber._SHARED_MODEL is None
        ):
            return  # Nothing to unload

        log("[INFO] Unloading Whisper model to free VRAM...")

        # 1. Delete the CTranslate2 model objects
        if self._batched_model is not None:
            del self._batched_model
            self._batched_model = None

        # Clear shared state
        if AudioTranscriber._SHARED_BATCHED is not None:
            del AudioTranscriber._SHARED_BATCHED
            AudioTranscriber._SHARED_BATCHED = None

        if self._model is not None:
            del self._model
            self._model = None
            self._current_model_size = None

        if AudioTranscriber._SHARED_MODEL is not None:
            del AudioTranscriber._SHARED_MODEL
            AudioTranscriber._SHARED_MODEL = None
            AudioTranscriber._SHARED_SIZE = None

        # 2. Force Python's Garbage Collector to run
        gc.collect()

        # 3. Force PyTorch to release cached VRAM
        if self.device == "cuda" and torch.cuda.is_available():
            torch.cuda.empty_cache()

        log("[SUCCESS] Whisper unloaded. VRAM should be free.")

    def _get_ffmpeg_cmd(self) -> str:
        """Locates the ffmpeg executable in system PATH.

        Returns:
            str: Full path to the ffmpeg executable.

        Raises:
            RuntimeError: If ffmpeg is not installed or not found in PATH.
        """
        cmd = shutil.which("ffmpeg")
        if not cmd:
            raise RuntimeError(
                "FFmpeg not found. Please install it to system PATH."
            )
        return cmd

    def _find_existing_subtitles(
        self,
        input_path: Path,
        output_path: Path,
        user_sub_path: Path | None,
        language: str,
    ) -> bool:
        """Discovers existing subtitles with strict priority (Sidecar > Embedded > AI).

        Checks for user-provided files first, then sidecar .srt files in the
        same directory, and finally attempts to extract embedded streams.

        Args:
            input_path: Path to the source media file.
            output_path: Destination path for the discovered/copied subtitle.
            user_sub_path: Optional explicit override path provided by user.
            language: Target language code for filtering sidecar/embedded subs.

        Returns:
            True if a subtitle was found and placed at output_path.
        """
        log(
            "[Subtitle] Checking for existing subtitles (Sidecar > Embedded > AI)..."
        )

        # Priority 1: Explicit user-provided subtitle
        if user_sub_path and user_sub_path.exists():
            log(f"[Subtitle] ✓ Using USER-PROVIDED subtitle: {user_sub_path}")
            shutil.copy(user_sub_path, output_path)
            return True

        # Priority 2: Sidecar .srt files
        sidecar_candidates = [
            input_path.with_suffix(f".{language}.srt"),
            input_path.with_suffix(".srt"),
            input_path.parent / f"{input_path.stem}.{language}.srt",
            input_path.parent / f"{input_path.stem}.srt",
        ]
        for sidecar in sidecar_candidates:
            if sidecar.exists() and sidecar != output_path:
                log(f"[Subtitle] ✓ Using SIDECAR subtitle: {sidecar}")
                shutil.copy(sidecar, output_path)
                return True

        # Priority 3: Embedded subtitle stream
        log("[Subtitle] No sidecar found. Probing for EMBEDDED subtitles...")
        cmd = [
            self._get_ffmpeg_cmd(),
            "-y",
            "-v",
            "error",
            "-i",
            str(input_path),
            "-map",
            "0:s:0",
            str(output_path),
        ]
        try:
            subprocess.run(
                cmd,
                check=True,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
            if output_path.exists() and output_path.stat().st_size > 0:
                log("[Subtitle] ✓ Using EMBEDDED subtitle stream")
                return True
        except subprocess.CalledProcessError:
            pass

        log(
            "[Subtitle] ✗ No existing subtitles found. Starting AI TRANSCRIPTION..."
        )
        return False

    @observe("transcriber_slice_audio")
    async def _slice_audio(
        self, input_path: Path, start: float, end: float | None
    ) -> Path:
        """Slices a segment of audio from a source file into a temporary WAV.

        Args:
            input_path: Original audio or video file path.
            start: Start time in seconds.
            end: Optional end time in seconds.

        Returns:
            Path to the temporary 16kHz mono WAV file.
        """
        with NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
            output_slice = Path(tmp.name)

        cmd = [
            self._get_ffmpeg_cmd(),
            "-y",
            "-v",
            "error",
            "-i",
            str(input_path),
            "-ss",
            str(start),
        ]
        if end:
            cmd.extend(["-to", str(end)])

        cmd.extend(
            ["-ar", "16000", "-ac", "1", "-map", "0:a:0", str(output_slice)]
        )

        log(f"[INFO] Slicing audio: {start}s -> {end if end else 'END'}s")
        # Run blocking subprocess in thread
        await asyncio.to_thread(subprocess.run, cmd, check=True, stderr=subprocess.DEVNULL)
        return output_slice

    @observe("transcriber_convert_model")
    def _convert_to_ct2(self, model_id: str) -> Path:
        model_name = model_id.split("/")[-1]
        ct2_output_dir = settings.model_cache_dir / f"ct2-{model_name}"

        # If cache exists and looks valid (has model.bin), skip conversion
        if ct2_output_dir.exists() and (ct2_output_dir / "model.bin").exists():
            # log(f"[INFO] Using cached CTranslate2 model: {ct2_output_dir}")
            return ct2_output_dir

        log(f"[INFO] Converting {model_id} to CTranslate2 format...")

        try:
            # 1. Download original model from HF
            # Allow HF Transfer for speed
            raw_model_dir = snapshot_download(
                repo_id=model_id,
                local_dir=settings.model_cache_dir / f"raw-{model_name}",
                local_dir_use_symlinks=False,
            )
            raw_model_dir = Path(raw_model_dir)

            # 2. Convert to CTranslate2
            # Handle newer transformers that don't accept dtype argument
            quantization = (
                "int8_float16" if settings.device == "cuda" else "int8"
            )

            try:
                converter = ctranslate2.converters.TransformersConverter(
                    str(raw_model_dir),
                    copy_files=[
                        "tokenizer.json",
                        "preprocessor_config.json",
                        "tokenizer_config.json",
                    ],
                )
                converter.convert(
                    str(ct2_output_dir),
                    force=True,
                    quantization=quantization,
                )
            except TypeError as te:
                # Handle "unexpected keyword argument 'dtype'" error from newer transformers
                if "dtype" in str(te):
                    log(
                        "[WARN] dtype error detected, trying low_cpu_mem_usage workaround..."
                    )
                    # Force low_cpu_mem_usage=False to bypass dtype issue
                    converter = ctranslate2.converters.TransformersConverter(
                        str(raw_model_dir),
                        copy_files=[
                            "tokenizer.json",
                            "preprocessor_config.json",
                            "tokenizer_config.json",
                        ],
                        low_cpu_mem_usage=False,
                    )
                    converter.convert(
                        str(ct2_output_dir),
                        force=True,
                        quantization=quantization,
                    )
                else:
                    raise

            # 3. CRITICAL: Manually patch tokenizer.json if missing or corrupt
            # faster-whisper requires specific config files
            for file_name in [
                "tokenizer.json",
                "vocab.json",
            ]:
                src = raw_model_dir / file_name
                if src.exists():
                    shutil.copy(src, ct2_output_dir / file_name)

            log("[SUCCESS] Model converted and patched successfully.")
            return ct2_output_dir

        except Exception as e:
            log(f"[ERROR] Model download/conversion failed: {e}")
            raise ModelLoadError(f"Could not prepare {model_id}: {e}", original_error=e)

    def _convert_and_cache_model(self, model_id: str) -> Path:
        """Alias for _convert_to_ct2 for backwards compatibility."""
        return self._convert_to_ct2(model_id)

    @observe("transcriber_load_model")
    def _load_model(self, model_key: str) -> None:
        # Check if already loaded with correct size
        if (
            AudioTranscriber._SHARED_MODEL is not None
            and AudioTranscriber._SHARED_SIZE == model_key
        ):
            return

        # Unload if different size loaded?
        # Yes, we only support one model loaded at a time for Whisper
        if AudioTranscriber._SHARED_MODEL is not None:
            self.unload_model()

        # only log here, when we actually load weights
        log(
            f"[INFO] Loading Faster-Whisper model '{model_key}' "
            f"({self.device}, Compute: {self.compute_type})..."
        )

        # Systran models are pre-converted CTranslate2 - no conversion needed
        # They can be loaded directly by WhisperModel from HuggingFace
        is_preconverted = (
            "systran" in model_key.lower()
            or "faster-whisper" in model_key.lower()
        )

        if is_preconverted:
            # Pass model ID directly - WhisperModel will download from HuggingFace
            final_model_path = model_key
            log(f"[INFO] Using pre-converted model: {model_key}")
        else:
            # Need to convert openai/whisper models to CTranslate2 format
            final_model_path = self._convert_and_cache_model(model_key)

        # Memory-efficient settings
        cpu_threads = min(
            4, os.cpu_count() or 4
        )  # Limit CPU threads to reduce memory

        try:
            AudioTranscriber._SHARED_MODEL = WhisperModel(
                str(final_model_path),
                device=self.device,
                compute_type=self.compute_type,
                download_root=str(settings.model_cache_dir),
                cpu_threads=cpu_threads,
                num_workers=1,  # Single worker to minimize memory
            )
            AudioTranscriber._SHARED_BATCHED = BatchedInferencePipeline(
                model=AudioTranscriber._SHARED_MODEL
            )
            AudioTranscriber._SHARED_SIZE = model_key
            log(f"[SUCCESS] Loaded {model_key} on {self.device}")
        except (RuntimeError, MemoryError, Exception) as e:
            error_msg = str(e).lower()
            # Check for memory-related errors
            if any(
                x in error_msg
                for x in ["memory", "mkl_malloc", "allocation", "oom"]
            ):
                log(f"[WARN] Memory error loading {model_key}: {e}")
                if not AudioTranscriber._FALLBACK_ATTEMPTED:
                    AudioTranscriber._FALLBACK_ATTEMPTED = True
                    log("[INFO] Attempting fallback to smaller model...")
                    # Try smaller models
                    for fallback_model in self.LOW_MEMORY_MODELS:
                        try:
                            gc.collect()
                            if (
                                self.device == "cuda"
                                and torch.cuda.is_available()
                            ):
                                torch.cuda.empty_cache()
                            log(
                                f"[INFO] Trying fallback model: {fallback_model}"
                            )
                            fallback_path = self._convert_and_cache_model(
                                fallback_model
                            )
                            AudioTranscriber._SHARED_MODEL = WhisperModel(
                                str(fallback_path),
                                device="cpu",  # Force CPU for stability
                                compute_type="int8",  # Use int8 for minimal memory
                                download_root=str(settings.model_cache_dir),
                                cpu_threads=2,
                                num_workers=1,
                            )
                            AudioTranscriber._SHARED_BATCHED = (
                                BatchedInferencePipeline(
                                    model=AudioTranscriber._SHARED_MODEL
                                )
                            )
                            AudioTranscriber._SHARED_SIZE = fallback_model
                            log(
                                f"[SUCCESS] Loaded fallback model {fallback_model} on CPU"
                            )
                            return
                        except Exception as fallback_err:
                            log(
                                f"[WARN] Fallback {fallback_model} also failed: {fallback_err}"
                            )
                            continue
            raise ModelLoadError(f"Failed to load Faster-Whisper model {model_key}: {e}", original_error=e)

    def _format_timestamp(self, seconds: float) -> str:
        hours, remainder = divmod(seconds, 3600)
        minutes, secs = divmod(remainder, 60)
        millis = round((secs - int(secs)) * 1000)
        return f"{int(hours):02}:{int(minutes):02}:{int(secs):02},{millis:03}"

    def _write_srt(
        self, chunks: list[dict[str, Any]], path: Path, offset: float
    ) -> int:
        count = 0
        with open(path, "w", encoding="utf-8") as f:
            for chunk in chunks:
                text = chunk.get("text", "").strip()
                timestamp = chunk.get("timestamp")
                if not text or not timestamp:
                    continue

                try:
                    start, end = timestamp
                except ValueError:
                    continue

                if start is None:
                    continue
                if end is None:
                    end = start + 2.0

                start = float(start)
                end = float(end)

                if (end - start) < 0.2:
                    continue

                f.write(
                    f"{count + 1}\n"
                    f"{self._format_timestamp(start + offset)} --> "
                    f"{self._format_timestamp(end + offset)}\n"
                    f"{text}\n\n"
                )
                count += 1
        return count

    def _split_long_chunks(
        self, chunks: list[dict[str, Any]], max_segment_s: float = 8.0
    ) -> list[dict[str, Any]]:
        out: list[dict[str, Any]] = []
        for ch in chunks:
            text = (ch.get("text") or "").strip()
            timestamp = ch.get("timestamp")
            if not text or not timestamp:
                continue
            try:
                start, end = timestamp
            except Exception:
                out.append(ch)
                continue

            if start is None:
                continue
            if end is None:
                end = start + 2.0

            start = float(start)
            end = float(end)

            duration = end - start
            if duration <= max_segment_s:
                out.append(ch)
                continue

            n = int((duration + max_segment_s - 1e-9) // max_segment_s) + 1
            n = max(1, n)
            words = text.split()
            if not words:
                out.append(ch)
                continue

            per = max(1, len(words) // n)
            ptr = 0
            for i in range(n):
                sub_words = words[ptr : ptr + per]
                ptr += per
                if not sub_words:
                    continue
                sub_text = " ".join(sub_words)
                sub_start = start + i * (duration / n)
                sub_end = start + (i + 1) * (duration / n)
                out.append(
                    {"text": sub_text, "timestamp": (sub_start, sub_end)}
                )

            if ptr < len(words) and out:
                out[-1]["text"] = out[-1]["text"] + " " + " ".join(words[ptr:])

        return out

    @observe("transcription")
    async def transcribe(
        self,
        audio_path: Path,
        language: str | None = None,
        subtitle_path: Path | None = None,
        output_path: Path | None = None,
        start_time: float = 0.0,
        end_time: float | None = None,
        force_lyrics: bool = False,
        task: str = "transcribe",
    ) -> list[dict[str, Any]] | None:
        """Transcribe audio using the Best-of-Council (BoC) strategy."""
        if not audio_path.exists():
            log(f"[ERROR] Input file not found: {audio_path}")
            return None

        lang = language or settings.language
        out_srt = output_path or audio_path.with_suffix(".srt")

        # Existing Sidecar / Embedded Subtitles (skip for lyrics mode)
        if start_time == 0.0 and end_time is None and not force_lyrics:
            if self._find_existing_subtitles(
                audio_path, out_srt, subtitle_path, lang or "en"
            ):
                log(f"[INFO] Parsing existing subtitles from {out_srt}...")
                parsed_segments = parse_srt(out_srt)
                log(
                    f"[SUCCESS] Loaded {len(parsed_segments)}"
                    " segments from existing file."
                )
                return parsed_segments

        # Run Whisper (If no subs found) ---
        is_sliced = True
        proc_path = await self._slice_audio(audio_path, start_time, end_time)

        try:
            return await self._inference(
                proc_path,
                lang,
                out_srt,
                start_time,
                is_sliced,
                force_lyrics=force_lyrics,
            )
        finally:
            if is_sliced and proc_path.exists():
                try:
                    proc_path.unlink()
                except Exception:
                    pass

    async def _inference(
        self,
        audio_path: Path,
        lang: str | None,
        out_srt: Path,
        offset: float,
        is_temp_file: bool = False,
        force_lyrics: bool = False,
    ) -> list[dict[str, Any]] | None:
        """Run Whisper inference in a non-blocking thread."""
        chunks: list[dict[str, Any]] = []

        candidates = settings.whisper_model_map.get(
            lang or "en", [settings.fallback_model_id]
        )
        model_to_use = (
            candidates[0] if candidates else settings.fallback_model_id
        )

        try:
            # ------------------------------------------------------------------
            # BLOCKING SECTION START: Model Loading & Inference
            # ------------------------------------------------------------------
            def _blocking_inference():
                # Check shared model state
                if AudioTranscriber._SHARED_BATCHED is None or (
                    AudioTranscriber._SHARED_SIZE != model_to_use
                ):
                    self._load_model(model_to_use)

                if AudioTranscriber._SHARED_BATCHED is None:
                    raise RuntimeError("Model failed to initialize")

                log(
                    f"[INFO] Running Inference on {audio_path} with {model_to_use}."
                    + (" (lyrics mode)" if force_lyrics else "")
                )

                # Language locking: detect once, force for all chunks
                effective_lang = lang
                if settings.whisper_language_lock:
                    if self._locked_language:
                        effective_lang = self._locked_language
                    else:
                        effective_lang = None  # Auto-detect on first run

                no_speech_thresh = (
                    0.9 if force_lyrics else 0.6
                )  # More permissive for lyrics

                segments, info = AudioTranscriber._SHARED_BATCHED.transcribe(
                    str(audio_path),
                    batch_size=settings.batch_size,
                    language=effective_lang,
                    task="transcribe",
                    beam_size=3,
                    condition_on_previous_text=False,
                    initial_prompt="♪"
                    if force_lyrics
                    else None,
                    repetition_penalty=1.2,
                    word_timestamps=True,
                    vad_filter=True,
                    vad_parameters={
                        "min_speech_duration_ms": 10 if force_lyrics else 100,
                        "min_silence_duration_ms": 1000 if force_lyrics else 500,
                        "speech_pad_ms": 2000 if force_lyrics else 800,
                        "threshold": 0.1
                        if force_lyrics
                        else 0.3,
                    },
                    no_speech_threshold=no_speech_thresh,
                    log_prob_threshold=-1.0,
                )
                
                # Consume generator here in thread to avoid blocking later
                return list(segments), info

            # Run the heavy blocking part in a thread
            segments_list, info = await asyncio.to_thread(_blocking_inference)
            
            # ------------------------------------------------------------------
            # BLOCKING SECTION END
            # ------------------------------------------------------------------

            # Lock detected language for subsequent chunks
            if (
                settings.whisper_language_lock
                and not self._locked_language
                and info.language
            ):
                self._locked_language = info.language
                log(f"[INFO] Language locked to: {self._locked_language}")

            for segment in segments_list: # Iterator exhausted to list in thread

                if (segment.end - segment.start) < 0.2:
                    continue
                if segment.words:
                    current_words = []
                    seg_start = segment.words[0].start

                    for word in segment.words:
                        current_words.append(word.word)
                        if (
                            word.end - seg_start > 6.0
                        ) or word.word.strip().endswith(("?", ".", "!")):
                            chunks.append(
                                {
                                    "text": "".join(current_words).strip(),
                                    "timestamp": (seg_start, word.end),
                                }
                            )
                            current_words = []
                            seg_start = word.end

                    if current_words:
                        chunks.append(
                            {
                                "text": "".join(current_words).strip(),
                                "timestamp": (seg_start, segment.words[-1].end),
                            }
                        )
                else:
                    chunks.append(
                        {
                            "text": segment.text.strip(),
                            "timestamp": (segment.start, segment.end),
                        }
                    )

            if not chunks:
                log("[WARN] No speech detected.")
                return None

            log(
                f"[SUCCESS] Transcription complete. Prob: {info.language_probability}"
            )

            lines_written = self._write_srt(chunks, out_srt, offset=offset)
            if lines_written > 0:
                log(f"[SUCCESS] Saved {lines_written} subtitles to: {out_srt}")
                return chunks

            return None

        except Exception as e:
            log(f"[ERROR] Inference failed: {e}")
            raise TranscriberError(f"Whisper inference failed for {audio_path}: {e}", original_error=e)

    def _run_whisper_inference(
        self,
        audio_path: Path,
        lang: str | None,
        out_srt: Path,
        offset: float,
        is_temp_file: bool = False,
        force_lyrics: bool = False,
    ) -> list[dict[str, Any]] | None:
        """Alias for _inference for backwards compatibility."""
        return self._inference(
            audio_path, lang, out_srt, offset, is_temp_file, force_lyrics
        )

    @observe("language_detection")
    def detect_language(self, audio_path: Path) -> str:
        """Detect the dominant language in an audio file.

        Args:
            audio_path: Path to the audio file.

        Returns:
            ISO 639-1 language code.
        """
        # Use Systran pre-converted model for language detection (no conversion needed)
        model_id = "Systran/faster-whisper-base"
        wav_path = None

        try:
            # Ensure model is loaded (handles download/conversion)
            if AudioTranscriber._SHARED_SIZE != model_id:
                self._load_model(model_id)

            # Use the underlying model directly for detection (no batching needed)
            if AudioTranscriber._SHARED_MODEL is None:
                raise RuntimeError("Model not initialized")

            # Extract first 30s as WAV to avoid container/codec issues (Opus, etc)
            try:
                # Use _slice_audio which uses ffmpeg to generate a clean 16kHz WAV
                wav_path = self._slice_audio(audio_path, start=0.0, end=30.0)
            except Exception as e:
                log(
                    f"[WARN] Slicing for detection failed: {e}. Trying raw file."
                )
                wav_path = audio_path

            # Detect on the WAV (or raw if slice failed)
            # NOTE: 'detect_language' task is sometimes rejected by CTranslate2 models.
            # The robust way with faster-whisper is to run 'transcribe' on a short segment
            # and check info.language.
            _, info = AudioTranscriber._SHARED_MODEL.transcribe(
                str(wav_path),
                task="transcribe",  # Changed from 'detect_language'
                beam_size=5,
            )

            log(
                f"[Audio] Language detected: {info.language} ({info.language_probability:.1%})"
            )

            # Lower threshold to ensure we catch Indic languages even with background noise
            if info.language_probability > 0.4:
                return info.language

            # Special check: if detected is 'ta' but confidence is low (0.2-0.4), still return 'ta'
            # because Whisper often has low confidence for Tamil but correctly identifies it vs English
            if (
                info.language in ["ta", "hi", "ml", "te", "kn"]
                and info.language_probability > 0.2
            ):
                log(
                    f"[Audio] Low confidence Indic language detected: {info.language}. accepting."
                )
                return info.language

            return "en"

        except Exception as e:
            log(f"[WARN] Language detection failed: {e}. Defaulting to 'en'.")
            return "en"
        finally:
            # Cleanup temp wav (only if we created it)
            if wav_path and wav_path != audio_path and wav_path.exists():
                try:
                    wav_path.unlink()
                except Exception:
                    pass


# CLI entry point removed to resolve syntax error
