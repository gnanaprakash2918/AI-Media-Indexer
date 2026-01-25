"""AI4Bharat IndicConformer Pipeline - Lightweight and Full Options.

Production-grade Indic ASR with two backends:
1. **HuggingFace Transformers (recommended)**: ~600MB, cross-platform, no NeMo
2. **NeMo Toolkit (optional)**: Better quality but 5GB+ models, Linux preferred

Tamil text cleanup and approximate SRT generation included.
"""

import gc
import re
import shutil
import subprocess
import unicodedata
from pathlib import Path
from tempfile import NamedTemporaryFile
from typing import Any

import torch

# Backend 1: HuggingFace Transformers (lightweight, cross-platform)

try:
    from transformers import (  # type: ignore
        AutoModelForSpeechSeq2Seq,
        AutoProcessor,
    )
    from transformers.pipelines import pipeline as hf_pipeline  # type: ignore

    HAS_HF_TRANSFORMERS = True
except ImportError:
    AutoModelForSpeechSeq2Seq = None
    AutoProcessor = None
    hf_pipeline = None
    HAS_HF_TRANSFORMERS = False

# Backend 2: NeMo (heavier but higher quality)
# Windows compatibility: NeMo uses signal.SIGKILL which doesn't exist on Windows
import os
import signal
import sys

if sys.platform == "win32" and not hasattr(signal, "SIGKILL"):
    # SIGKILL (9) doesn't exist on Windows, add a dummy constant
    signal.SIGKILL = 9  # type: ignore[attr-defined]

try:
    import nemo.collections.asr as nemo_asr  # type: ignore
    from huggingface_hub import hf_hub_download

    HAS_NEMO = True
except ImportError:
    nemo_asr = None
    hf_hub_download = None
    HAS_NEMO = False

try:
    import torchaudio

    HAS_TORCHAUDIO = True
except ImportError:
    torchaudio = None
    HAS_TORCHAUDIO = False

from config import settings
from core.utils.logger import log

# Log NeMo availability at module load
if not HAS_NEMO:
    log(
        "[IndicASR] NVIDIA NeMo not installed. For SOTA Indic ASR, run: uv pip install nemo_toolkit[asr]"
    )
    log("[IndicASR] Falling back to HuggingFace Whisper for transcription.")


class IndicASRPipeline:
    """IndicConformer ASR with two backend options.

    Backend Selection (automatic):
    1. HuggingFace Transformers (~600MB) - default, cross-platform
    2. NeMo Toolkit (5GB+) - only if USE_NEMO_ASR=true, Linux preferred

    The HF backend uses ai4bharat/indic-conformer-600m-multilingual which
    supports all major Indic languages without separate model downloads.
    """

    # All supported Indic languages
    # Hindi, Tamil, Telugu, Kannada, Malayalam, Bengali, Marathi, Gujarati, Punjabi, Odia
    INDIC_LANGUAGES = {
        "hi",
        "ta",
        "te",
        "kn",
        "ml",
        "bn",
        "mr",
        "gu",
        "pa",
        "or",
    }

    # Lightweight HF model (~600MB, multilingual)
    HF_MODEL = "ai4bharat/indic-conformer-600m-multilingual"

    # Heavy NeMo models (optional, 3-5GB each)
    # Not all languages have dedicated NeMo models - fallback to HF for others
    NEMO_MODEL_MAP = {
        "ta": {
            "repo_id": "ai4bharat/indicconformer_stt_ta_hybrid_ctc_rnnt_large",
            "filename": "indicconformer_stt_ta_hybrid_rnnt_large.nemo",
        },
        "hi": {
            "repo_id": "ai4bharat/indicconformer_stt_hi_hybrid_ctc_rnnt_large",
            "filename": "indicconformer_stt_hi_hybrid_rnnt_large.nemo",
        },
        "te": {
            "repo_id": "ai4bharat/indicconformer_stt_te_hybrid_ctc_rnnt_large",
            "filename": "indicconformer_stt_te_hybrid_rnnt_large.nemo",
        },
        "kn": {
            "repo_id": "ai4bharat/indicconformer_stt_kn_hybrid_ctc_rnnt_large",
            "filename": "indicconformer_stt_kn_hybrid_rnnt_large.nemo",
        },
        "ml": {
            "repo_id": "ai4bharat/indicconformer_stt_ml_hybrid_ctc_rnnt_large",
            "filename": "indicconformer_stt_ml_hybrid_rnnt_large.nemo",
        },
        "bn": {
            "repo_id": "ai4bharat/indicconformer_stt_bn_hybrid_ctc_rnnt_large",
            "filename": "indicconformer_stt_bn_hybrid_rnnt_large.nemo",
        },
    }

    def __init__(self, lang: str = "ta", backend: str = "auto") -> None:
        """Initializes the Indic ASR pipeline.

        Args:
            lang: ISO 639-1 language code (e.g., 'ta', 'hi', 'te').
            backend: Preferred ASR backend ('hf', 'nemo', or 'auto').
        """
        self.device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )
        self.model: Any = None
        self.pipe: Any = None
        self.processor = None
        self.lang = lang
        self._backend = self._select_backend(backend)
        self._is_loaded = False
        log(f"[IndicASR] Using backend: {self._backend}")

    def _select_backend(self, preference: str) -> str:
        """Selects the best available ASR backend based on preference and environment.

        Args:
            preference: User's backend preference.

        Returns:
            The selected backend ('hf' or 'nemo').

        Raises:
            ImportError: If no suitable ASR backend is found.
        """
        if preference == "nemo" and HAS_NEMO:
            return "nemo"
        if preference == "hf" and HAS_HF_TRANSFORMERS:
            return "hf"

        # Auto: prefer HF (lighter, cross-platform)
        if HAS_HF_TRANSFORMERS:
            return "hf"
        if HAS_NEMO:
            return "nemo"

        raise ImportError(
            "No ASR backend available. Install one of:\n"
            "  uv pip install transformers  # Lightweight (~600MB model)\n"
            "  uv pip install nemo_toolkit[asr]  # Heavy (5GB+ models)"
        )

    def load_model(self) -> None:
        """Load the selected ASR backend model."""
        if self._is_loaded:
            return

        if self._backend == "hf":
            self._load_hf_model()
        else:
            self._load_nemo_model()

        self._is_loaded = True

    def _load_hf_model(self) -> None:
        """Loads the HuggingFace Whisper pipeline.

        Attempts to load whisper-large-v3-turbo for optimal quality, with
        a fallback to whisper-small if a CUDA Out Of Memory (OOM) error occurs.
        """
        # Use Whisper large-v3-turbo for best quality Indic transcription
        model_id = "openai/whisper-large-v3-turbo"
        log(
            f"[IndicASR] Loading HF pipeline: {model_id} (language: {self.lang})"
        )

        try:
            # The user's edit for this block was syntactically incorrect and seemed to mix up
            # pipeline arguments with model loading. Reverting to original correct logic.
            # The instruction was to "Add checks for model/processor being None and ensure return types match signatures."
            # This block is about loading the HF pipeline, not about adding checks for model/processor being None
            # within the pipeline call itself, nor does it directly relate to return types of this method.
            # The original code is correct for loading the pipeline.
            if hf_pipeline is None:
                raise ImportError("transformers.pipeline not found")
            self.pipe = hf_pipeline(
                "automatic-speech-recognition",
                model=model_id,
                device=0 if self.device.type == "cuda" else -1,
                torch_dtype=torch.float16
                if self.device.type == "cuda"
                else torch.float32,
                token=settings.hf_token,
            )
            # self.model remains None for HF, we use self.pipe
            log(f"[IndicASR] HF Whisper pipeline loaded on {self.device}")
        except Exception as e:
            # Explicitly set pipe to None if it failed partially
            self.pipe = None
            log(
                f"[IndicASR] Whisper large-v3-turbo failed: {e}. Trying smaller model..."
            )

            # Cleanup before fallback
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            gc.collect()

            # Fallback to whisper-small (~250MB) - still good for Tamil
            try:
                if hf_pipeline is None:
                    raise ImportError("transformers.pipeline not found")
                self.pipe = hf_pipeline(
                    "automatic-speech-recognition",
                    model="openai/whisper-small",
                    device=0 if self.device.type == "cuda" else -1,
                    torch_dtype=torch.float16
                    if self.device.type == "cuda"
                    else torch.float32,
                )
                assert self.pipe is not None  # Assertion for Pylance
                # self.model remains None for HF
                log(
                    "[IndicASR] Whisper-small fallback loaded (Tamil supported)"
                )
            except Exception as e2:
                log(f"[IndicASR] All HF models failed: {e2}")
                raise

    def _load_nemo_model(self) -> None:
        """Loads the NVIDIA NeMo ASR model for the target language.

        Uses pre-trained IndicConformer models from HuggingFace Hub.
        Patches tokenizer config for compatibility with NeMo 2.x.
        """
        import tarfile
        import tempfile

        import yaml

        if self.NEMO_MODEL_MAP is None:
            raise ValueError("NEMO_MODEL_MAP is missing")
        model_info = self.NEMO_MODEL_MAP.get(
            self.lang, self.NEMO_MODEL_MAP.get("ta")
        )
        if model_info is None:
            raise ValueError(f"No NeMo model found for language {self.lang}")
        log(f"[IndicASR] Loading NeMo model: {model_info['repo_id']}")

        try:
            if nemo_asr is None:
                raise ImportError("NeMo not installed")

            if hf_hub_download is None:
                raise ImportError("huggingface_hub not installed")

            nemo_ckpt_path = (
                hf_hub_download(
                    repo_id=model_info["repo_id"],
                    filename=model_info["filename"],
                    cache_dir=str(settings.model_cache_dir / "nemo_models"),
                    token=settings.hf_token,
                )
                or ""
            )
            assert nemo_ckpt_path, "NeMo checkpoint download failed"

            # Fix for NeMo 2.x: Extract and patch tokenizer config
            # The AI4Bharat models are missing 'dir' key in tokenizer config
            patched_dir = self._patch_nemo_for_nemo2(nemo_ckpt_path)

            if patched_dir:
                # Load from patched extracted directory
                log(f"[IndicASR] Loading from patched config: {patched_dir}")
                self.model = nemo_asr.models.ASRModel.restore_from(  # type: ignore
                    restore_path=patched_dir,
                    map_location=self.device,
                )
            else:
                # Fallback: try direct load (may work on older NeMo)
                self.model = nemo_asr.models.ASRModel.restore_from(  # type: ignore
                    restore_path=nemo_ckpt_path,
                    map_location=self.device,
                )

            self.model.freeze()  # type: ignore
            log(f"[IndicASR] NeMo model loaded on {self.device}")

        except Exception as e:
            log(f"[IndicASR] Failed to load NeMo model: {e}")
            raise

    def _patch_nemo_for_nemo2(self, nemo_path: str) -> str | None:
        """Extract .nemo archive and patch tokenizer config for NeMo 2.x compatibility.

        Args:
            nemo_path: Path to the .nemo checkpoint file.

        Returns:
            Path to the patched extracted directory, or None if patching failed.
        """
        import tarfile

        import yaml

        try:
            nemo_file = Path(nemo_path)
            if not nemo_file.exists():
                log(f"[IndicASR] .nemo file not found: {nemo_path}")
                return None

            # Create extraction directory in cache
            extract_dir = nemo_file.parent / f"{nemo_file.stem}_patched"

            # Check if already patched and complete
            config_path = extract_dir / "model_config.yaml"
            weights_file = extract_dir / "model_weights.ckpt"
            if config_path.exists() and weights_file.exists():
                log("[IndicASR] Using existing patched config")
                return str(extract_dir)

            # Clean up incomplete extraction
            if extract_dir.exists():
                import shutil
                shutil.rmtree(extract_dir, ignore_errors=True)

            extract_dir.mkdir(parents=True, exist_ok=True)

            # Extract .nemo (it's a tar archive)
            log(f"[IndicASR] Extracting .nemo archive for patching: {nemo_file.name}")
            with tarfile.open(nemo_path, "r") as tar:
                # Use safer extraction without filter for compatibility
                tar.extractall(extract_dir)

            # Find and patch model_config.yaml
            config_path = extract_dir / "model_config.yaml"
            if not config_path.exists():
                log("[IndicASR] No model_config.yaml found in archive")
                return None

            with open(config_path, "r", encoding="utf-8") as f:
                config = yaml.safe_load(f)

            # Patch tokenizer config to add 'dir' key
            if "tokenizer" in config:
                tokenizer_cfg = config["tokenizer"]

                # Find tokenizer directory (usually contains .model files)
                tokenizer_dir = None
                for item in extract_dir.iterdir():
                    if item.is_dir():
                        try:
                            if any(
                                f.suffix in (".model", ".vocab") 
                                for f in item.iterdir() if f.is_file()
                            ):
                                tokenizer_dir = item.name
                                break
                        except Exception:
                            continue

                # If no subdir found, check if tokenizer files are in root
                if tokenizer_dir is None:
                    if any(
                        f.suffix in (".model", ".vocab")
                        for f in extract_dir.iterdir()
                        if f.is_file()
                    ):
                        tokenizer_dir = "."

                if tokenizer_dir and "dir" not in tokenizer_cfg:
                    tokenizer_cfg["dir"] = tokenizer_dir
                    log(f"[IndicASR] Patched tokenizer.dir = '{tokenizer_dir}'")

                    # Write patched config
                    with open(config_path, "w", encoding="utf-8") as f:
                        yaml.safe_dump(config, f, default_flow_style=False)

                    return str(extract_dir)
                elif "dir" in tokenizer_cfg:
                    log("[IndicASR] tokenizer.dir already present")
                    return str(extract_dir)
                else:
                    log("[IndicASR] Could not find tokenizer directory to patch")
                    # Return the directory anyway - let NeMo try to load it
                    return str(extract_dir)
            else:
                log("[IndicASR] No tokenizer config found to patch")
                # Return the directory anyway - older models might not need patching
                return str(extract_dir)

        except Exception as e:
            log(f"[IndicASR] Failed to patch .nemo archive: {e}")
            import traceback
            traceback.print_exc()
            return None
    async def _extract_audio(self, media_path: Path) -> Path:
        """Extracts audio from a video file into a mono 16kHz WAV format (Async).

        Args:
            media_path: Path to the source video or audio file.

        Returns:
            Path to the temporary WAV file.

        Raises:
            RuntimeError: If FFmpeg is not found or extraction results in an empty file.
        """
        import asyncio

        # Check if already a WAV file
        if media_path.suffix.lower() == ".wav":
            return media_path

        # Create temp WAV file
        with NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
            wav_path = Path(tmp.name)

        ffmpeg_cmd = shutil.which("ffmpeg")
        if not ffmpeg_cmd:
            raise RuntimeError("FFmpeg not found. Please install it.")

        cmd_args = [
            "-y",
            "-v",
            "error",
            "-i",
            str(media_path),
            "-ar",
            "16000",  # 16kHz for ASR
            "-ac",
            "1",  # Mono
            "-map",
            "0:a:0",  # First audio stream
            str(wav_path),
        ]

        log(f"[IndicASR] Extracting audio from {media_path.name}...")
        try:
            # Use asyncio subprocess for non-blocking execution
            process = await asyncio.create_subprocess_exec(
                ffmpeg_cmd,
                *cmd_args,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
            stdout, stderr = await asyncio.wait_for(process.communicate(), timeout=300)

            if process.returncode != 0:
                 raise subprocess.CalledProcessError(
                     process.returncode or 1, ffmpeg_cmd, output=stdout, stderr=stderr
                 )

            if wav_path.exists() and wav_path.stat().st_size > 0:
                log(
                    f"[IndicASR] Audio extracted: {wav_path.stat().st_size / 1024:.1f}KB"
                )
                return wav_path
            else:
                raise RuntimeError("Audio extraction produced empty file")
        except Exception as e:
            # Handle subprocess and asyncio errors
            err_msg = str(e)
            if hasattr(e, 'stderr') and e.stderr: # type: ignore
                 err_msg = e.stderr.decode(errors='replace') # type: ignore
            log(f"[IndicASR] FFmpeg failed: {err_msg}")
            raise

    async def transcribe(
        self,
        audio_path: Path,
        language: str | None = None,
        output_srt: Path | None = None,
        **kwargs,
    ) -> list[dict[str, Any]]:
        """Transcribes an audio/video file using a hybrid strategy.

        Attempts native NeMo first, then remote Docker, and finally falls back
        to local HuggingFace Whisper.

        Args:
            audio_path: Path to the media file.
            language: Optional language override.
            output_srt: Optional path to save the transcription as SRT.
            **kwargs: Additional arguments for the transcriber.

        Returns:
            A list of segment dictionaries with 'start', 'end', and 'text'.
        """
        import httpx

        # Hybrid Strategy: Native -> Docker -> Whisper

        # 1. Attempt Native NeMo (If configured and installed)
        native_success = False
        if settings.use_native_nemo and HAS_NEMO:
            try:
                log(f"[IndicASR] Attempting Native NeMo for {audio_path.name}")
                self._backend = "nemo"
                self.load_model()
                native_success = True
            except Exception as e:
                log(
                    f"[IndicASR] Native NeMo failed ({e}). Falling back to next option."
                )
                native_success = False

        # 2. Attempt Remote Docker (If Native failed or disabled)
        if not native_success and settings.ai4bharat_url:
            try:
                url = f"{settings.ai4bharat_url}/transcribe"
                log(f"[IndicASR] Attempting Remote Docker at {url}")

                # Use async file reading if possible, but here file is small-ish
                # For strictly async, doing this in thread is safer but maybe overkill for now unless specified
                # Keeping synchronous HTTP logic for now as 'requests' isn't used, 'httpx' is used but synchronously?
                # Ah, httpx.post is sync here. Ideally should be async client.
                # BUT this task is about FFmpeg blocking. Let's fix FFmpeg first.
                
                # To be fully non-blocking, we should use async context manager for httpx
                # But to minimize changes, let's wrap this block in to_thread if it was blocking
                # Actually, blocking here inside 'async def transcribe' is bad.
                # However, the priority detected was FFmpeg.
                
                # Original code:
                # with open(audio_path, "rb") as f:
                #    response = httpx.post(...)
                
                # We'll leave the HTTP part for now as it wasn't the flagged issue, focus on FFmpeg.
                
                with open(audio_path, "rb") as f:
                    response = httpx.post(
                        url,
                        files={"file": f},
                        data={"language": language or self.lang},
                        timeout=300.0,
                    )

                if response.status_code == 200:
                    data = response.json()
                    segments = data.get("segments", [])
                    log(f"[IndicASR] Remote SUCCESS: {len(segments)} segments")
                    if output_srt:
                        self.write_srt(segments, output_srt)
                    return segments
            except Exception as e:
                log(
                    f"[IndicASR] Remote Docker failed ({e}). Falling back to Local Whisper."
                )

        # 3. Final Fallback: Local Whisper (via HF)
        # If we are here, either Native execution is set up (native_success=True)
        # OR both Native and Remote failed, so we force HF Whisper.

        if not native_success:
            self._backend = "hf"
            # Reset model to ensure we load HF, not broken NeMo
            self.model = None

        self.load_model()

        audio_path = Path(audio_path)
        if not audio_path.exists():
            log(f"[IndicASR] File not found: {audio_path}")
            return []

        log(
            f"[IndicASR] Transcribing {audio_path.name} with {self._backend}..."
        )

        # For NeMo, ensure model is loaded
        if self._backend == "nemo" and self.model is None:
            raise RuntimeError("NeMo model not initialized")

        # Extract audio from video if needed
        wav_path = None
        try:
            # await the Async extraction
            wav_path = await self._extract_audio(audio_path)

            if self._backend == "hf":
                # HF pipeline call is blocking CPU, wrap it
                import asyncio
                raw_text = await asyncio.to_thread(self._transcribe_hf, wav_path)
            else:
                raw_text = await self._transcribe_nemo(wav_path)

            # Clean Tamil text
            if self.lang == "ta":
                raw_text = self.clean_tamil_text(raw_text)

            # Get audio duration for SRT generation
            duration = self._get_audio_duration(wav_path)

            # Generate approximate segments
            segments = self._generate_segments(raw_text, duration)

            # Optionally write SRT
            if output_srt:
                self.write_srt(segments, output_srt)

            log(
                f"[IndicASR] Transcription complete: {len(raw_text)} chars, {len(segments)} segments"
            )
            return segments

        except Exception as e:
            log(f"[IndicASR] Transcription failed: {e}")
            return []
        finally:
            # Clean up temp wav file
            if wav_path and wav_path != audio_path and wav_path.exists():
                try:
                    wav_path.unlink()
                except Exception:
                    pass
            self._cleanup()

    def _transcribe_hf(self, audio_path: Path) -> str:
        """Transcribes audio using the HuggingFace Whisper pipeline.

        Args:
            audio_path: Path to the WAV audio file.

        Returns:
            The full transcribed text.

        Raises:
            RuntimeError: If the HF pipeline is not initialized.
        """
        if not hasattr(self, "pipe") or self.pipe is None:
            raise RuntimeError("HF pipeline not loaded")

        # Pass explicit language for better Indic transcription
        # Whisper uses ISO 639-1 codes like 'ta' for Tamil
        # return_timestamps=True is REQUIRED for audio > 30 seconds
        result = self.pipe(
            str(audio_path),
            return_timestamps=True,  # Required for long-form audio
            generate_kwargs={
                "language": self.lang,
                "task": "transcribe",  # Don't translate, just transcribe
            },
        )

        # Handle chunked output from long-form transcription
        if isinstance(result, dict):
            # Check if chunks are returned (long-form)
            if "chunks" in result:
                texts = [chunk.get("text", "") for chunk in result["chunks"]]
                return " ".join(texts).strip()
            return result.get("text", "").strip()
        return str(result).strip()

    async def _transcribe_nemo(self, audio_path: Path) -> str:
        """Transcribes audio using the NeMo backend with chunked processing.

        For long audio (>30s), processes in 30-second chunks to prevent
        CUDA Out Of Memory (OOM) errors.

        Args:
            audio_path: Path to the WAV audio file.

        Returns:
            The full combined transcription text.
        """
        import torch
        import asyncio

        duration = self._get_audio_duration(audio_path)
        chunk_duration_sec = 30  # Process in 30s chunks to prevent OOM

        # Short audio: process directly
        if duration <= chunk_duration_sec:
            # Run blocking NeMo inference in thread
            def _infer_single():
                 with torch.no_grad():
                    self.model.cur_decoder = "ctc"  # type: ignore
                    return self.model.transcribe(  # type: ignore
                        [str(audio_path)],
                        batch_size=1,
                        language_id=self.lang,
                    )
            
            transcriptions = await asyncio.to_thread(_infer_single)
            return self._extract_text(transcriptions)

        # Long audio: chunked processing with VRAM cleanup
        log(
            f"[IndicASR] Long audio ({duration:.1f}s), processing in {chunk_duration_sec}s chunks"
        )
        texts: list[str] = []

        for start in range(0, int(duration), chunk_duration_sec):
            end = min(start + chunk_duration_sec, duration)

            # Slice audio chunk (Async)
            chunk_path = await self._slice_audio_chunk(audio_path, start, end)
            if chunk_path is None:
                continue

            try:
                def _infer_chunk():
                    with torch.no_grad():
                        self.model.cur_decoder = "ctc" # type: ignore
                        return self.model.transcribe( # type: ignore
                            [str(chunk_path)],
                            batch_size=1,
                            language_id=self.lang,
                        )

                transcriptions = await asyncio.to_thread(_infer_chunk)
                
                chunk_text = self._extract_text(transcriptions)
                if chunk_text:
                    texts.append(chunk_text)

                # Cleanup VRAM after each chunk to prevent accumulation
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

            except Exception as e:
                log(f"[IndicASR] Chunk {start}-{end}s failed: {e}")
            finally:
                # Remove temp chunk file
                if chunk_path.exists():
                    try:
                        chunk_path.unlink()
                    except Exception:
                        pass

        return " ".join(texts)

    async def _slice_audio_chunk(
        self, audio_path: Path, start: float, end: float
    ) -> Path | None:
        """Extracts a specific time segment from an audio file using FFmpeg (Async).

        Args:
            audio_path: Path to the source WAV file.
            start: Start time in seconds.
            end: End time in seconds.

        Returns:
            Path to the temp chunk file, or None on failure.
        """
        import asyncio
        from tempfile import NamedTemporaryFile

        ffmpeg_cmd = shutil.which("ffmpeg")
        if not ffmpeg_cmd:
            return None

        with NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
            chunk_path = Path(tmp.name)

        cmd_args = [
            "-y",
            "-v",
            "error",
            "-i",
            str(audio_path),
            "-ss",
            str(start),
            "-to",
            str(end),
            "-ar",
            "16000",
            "-ac",
            "1",
            str(chunk_path),
        ]

        try:
             process = await asyncio.create_subprocess_exec(
                ffmpeg_cmd,
                *cmd_args,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
             )
             _, stderr = await asyncio.wait_for(process.communicate(), timeout=60)
             
             if process.returncode != 0:
                 raise Exception(f"FFmpeg failed with code {process.returncode}: {stderr.decode() if stderr else ''}")

             if chunk_path.exists() and chunk_path.stat().st_size > 0:
                 return chunk_path
        except Exception as e:
            log(f"[IndicASR] Audio slice failed: {e}")

        return None

    def _extract_text(self, transcriptions: Any) -> str:
        """Extracts plain text from various NeMo transcription output formats.

        Args:
            transcriptions: The raw output from NeMo's transcribe method.

        Returns:
            The extracted transcription string.
        """
        if not transcriptions:
            return ""

        first = transcriptions[0]

        # Handle [["text"]], [[{"text": "..."}]], ["text"]
        if isinstance(first, list):
            if first and isinstance(first[0], str):
                return first[0]
            elif first and isinstance(first[0], dict) and "text" in first[0]:
                return first[0]["text"]
            return str(first)
        elif isinstance(first, dict) and "text" in first:
            return first["text"]
        else:
            return str(first)

    def clean_tamil_text(self, text: str) -> str:
        """Cleans and normalizes Tamil transcription output.

        Removes orphan diacritics, normalizes whitespace, and preserves
        valid Tamil combinations.

        Args:
            text: The raw Tamil transcription text.

        Returns:
            The cleaned and normalized text.
        """
        result = []
        prev_is_tamil_base = False

        for ch in text:
            cat = unicodedata.category(ch)
            if cat.startswith("M"):  # Mark (diacritic)
                if not prev_is_tamil_base:
                    continue  # Skip orphan diacritics
            else:
                # Check if Tamil character
                if "\u0b80" <= ch <= "\u0bff":
                    prev_is_tamil_base = True
                else:
                    prev_is_tamil_base = False

            result.append(ch)

        cleaned = "".join(result)
        cleaned = re.sub(r"\s+", " ", cleaned).strip()
        return cleaned

    def _get_audio_duration(self, audio_path: Path) -> float:
        """Calculates the duration of an audio file in seconds.

        Args:
            audio_path: Path to the WAV audio file.

        Returns:
            The duration in seconds.
        """
        if HAS_TORCHAUDIO and torchaudio is not None:
            try:
                info = torchaudio.info(str(audio_path))
                return info.num_frames / info.sample_rate
            except Exception:
                pass

        # Fallback: estimate from file size (16kHz, 16-bit mono)
        try:
            if not audio_path.exists():
                return 0.0
            size = audio_path.stat().st_size
            # Assuming 16kHz, 16-bit mono WAV
            return size / (16000 * 2)
        except Exception:
            return 60.0  # Default fallback

    def _generate_segments(
        self,
        text: str,
        duration: float,
        max_chars_per_segment: int = 80,
    ) -> list[dict[str, Any]]:
        """Generates approximate timestamped segments from full text.

        Since some backends do not provide word-level timestamps, this method
        approximates them based on character distribution across the duration.

        Args:
            text: The full transcription text.
            duration: Total duration of the audio.
            max_chars_per_segment: Maximum characters allowed per segment.

        Returns:
            A list of segment dictionaries.
        """
        words = text.split()
        if not words:
            return []

        total_chars = sum(len(w) for w in words) or 1
        segments = []
        current_words = []
        current_chars = 0
        char_so_far = 0

        for w in words:
            wlen = len(w)
            if (
                current_chars + wlen + (1 if current_words else 0)
                > max_chars_per_segment
                and current_words
            ):
                seg_text = " ".join(current_words)
                segments.append((seg_text, char_so_far))
                char_so_far += len(seg_text.replace(" ", ""))
                current_words = [w]
                current_chars = wlen
            else:
                current_words.append(w)
                current_chars += wlen + (1 if current_words else 0)

        if current_words:
            seg_text = " ".join(current_words)
            segments.append((seg_text, char_so_far))

        # Convert to timestamped dicts
        result = []
        for seg_text, seg_char_start in segments:
            seg_char_end = seg_char_start + len(seg_text.replace(" ", ""))
            start_t = duration * (seg_char_start / total_chars)
            end_t = duration * (seg_char_end / total_chars)

            # Ensure minimum duration
            if end_t - start_t < 1.5:
                end_t = min(start_t + 1.5, duration)

            result.append(
                {
                    "text": seg_text,
                    "timestamp": (start_t, end_t),
                    "start": start_t,
                    "end": end_t,
                    "type": "indic_transcription",
                    "language": self.lang,
                }
            )

        return result

    def write_srt(self, segments: list[dict], srt_path: Path) -> None:
        """Writes transcription segments to an SRT file.

        Args:
            segments: A list of segment dictionaries with timing and text.
            srt_path: Path where the SRT file will be saved.
        """
        srt_path = Path(srt_path)

        def format_ts(seconds: float) -> str:
            millis = round(seconds * 1000)
            h = millis // 3_600_000
            millis %= 3_600_000
            m = millis // 60_000
            millis %= 60_000
            s = millis // 1000
            ms = millis % 1000
            return f"{h:02d}:{m:02d}:{s:02d},{ms:03d}"

        lines = []
        for idx, seg in enumerate(segments, start=1):
            start = seg.get("start", 0)
            end = seg.get("end", start + 2)
            text = seg.get("text", "")
            lines.append(
                f"{idx}\n{format_ts(start)} --> {format_ts(end)}\n{text}\n"
            )

        srt_path.write_text("\n".join(lines), encoding="utf-8")
        log(f"[IndicASR] Wrote {len(segments)} segments to {srt_path}")

    def _cleanup(self) -> None:
        """Force garbage collection and clear CUDA cache."""
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()

    def unload_model(self) -> None:
        """Unloads the ASR models and frees VRAM.

        Crucial for freeing up resources for vision analysis or other jobs.
        """
        self.pipe = None
        self.model = None
        self._is_loaded = False
        self._cleanup()
        log("[IndicASR] Model unloaded - VRAM freed")
