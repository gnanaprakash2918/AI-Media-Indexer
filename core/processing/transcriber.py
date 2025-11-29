"""Robust Audio Transcriber with Subtitle Fallback and RTX Optimization.

Features:
- Smart Caching: Project Local > Global Cache > Download.
- Self-Healing: Automatically retries next model if one fails or returns empty data.
- Prioritizes existing subtitles (User provided > Sidecar > Embedded).
- Optimizes for RTX 40-series using SDPA (Scaled Dot Product Attention).
"""

import gc
import math
import shutil
import subprocess
import sys
import warnings
from pathlib import Path
from tempfile import NamedTemporaryFile
from typing import Any

import torch
from huggingface_hub import snapshot_download
from transformers import (
    AutoModelForSpeechSeq2Seq,
    AutoProcessor,
    AutoTokenizer,
)
from transformers.generation.configuration_utils import GenerationConfig
from transformers.pipelines import pipeline
from transformers.pipelines.base import Pipeline
from transformers.utils import logging as hf_logging

from config import settings

# Cleanup console output
warnings.filterwarnings("ignore")
hf_logging.set_verbosity_error()


class AudioTranscriber:
    """Main transcription class handling ASR lifecycle and subtitle extraction."""

    def __init__(self) -> None:
        """Initialize the transcriber state."""
        self._pipe: Pipeline | None = None
        self._current_model_id: str | None = None
        print(
            f"[INFO] Initialized Transcriber (Device: {settings.device}, "
            f"Dtype: {settings.torch_dtype})"
        )

    def _get_ffmpeg_cmd(self) -> str:
        """Locates FFmpeg executable."""
        cmd = shutil.which("ffmpeg")
        if not cmd:
            raise RuntimeError("FFmpeg not found. Please install it to system PATH.")
        return cmd

    def _find_existing_subtitles(
        self,
        input_path: Path,
        output_path: Path,
        user_sub_path: Path | None,
        language: str,
    ) -> bool:
        """Probes for existing subtitles in user input, sidecars, embedded streams."""
        print("[INFO] Probing for existing subtitles...")

        # 1. User Provided
        if user_sub_path and user_sub_path.exists():
            print(f"[SUCCESS] Using provided subtitle: {user_sub_path}")
            shutil.copy(user_sub_path, output_path)
            return True

        # 2. Sidecar Files
        for sidecar in [
            input_path.with_suffix(f".{language}.srt"),
            input_path.with_suffix(".srt"),
        ]:
            if sidecar.exists() and sidecar != output_path:
                print(f"[SUCCESS] Found sidecar: {sidecar}")
                shutil.copy(sidecar, output_path)
                return True

        # 3. Embedded Stream
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
                cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL
            )
            if output_path.exists() and output_path.stat().st_size > 0:
                print("[SUCCESS] Extracted embedded subtitles.")
                return True
        except subprocess.CalledProcessError:
            pass

        print("[INFO] No existing subtitles found. Proceeding to ASR.")
        return False

    def _slice_audio(self, input_path: Path, start: float, end: float | None) -> Path:
        """Slices audio safely using FFmpeg into a temp file."""
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

        cmd.extend(["-ar", "16000", "-ac", "1", "-map", "0:a:0", str(output_slice)])

        print(f"[INFO] Slicing audio: {start}s -> {end if end else 'END'}s")
        subprocess.run(cmd, check=True, stderr=subprocess.DEVNULL)
        return output_slice

    def _resolve_model_path(self, model_id: str) -> str:
        """Resolves model path: Local Project -> Global Cache -> Download."""

        def _try_download(
            local_only: bool = False, cache_dir: Path | None = None
        ) -> Path:
            return Path(
                snapshot_download(
                    repo_id=model_id,
                    cache_dir=cache_dir,
                    token=settings.hf_token,
                    local_files_only=local_only,
                    ignore_patterns=["*.msgpack", "*.h5", "*.ot"],
                )
            )

        # 1. Check Project Cache
        try:
            path = _try_download(local_only=True, cache_dir=settings.model_cache_dir)
            print(f"[INFO] Found '{model_id}' in project cache.")
            return str(path)
        except Exception:
            pass

        # 2. Check Global Cache
        try:
            print(f"[INFO] Checking global cache for '{model_id}'...")
            path = _try_download(local_only=True, cache_dir=None)
            print(f"[INFO] Found '{model_id}' in global cache.")
            return str(path)
        except Exception:
            pass

        # 3. Download to Project Cache
        print(f"[INFO] Downloading '{model_id}' to '{settings.model_cache_dir}'...")
        try:
            path = _try_download(local_only=False, cache_dir=settings.model_cache_dir)
            return str(path)
        except Exception as e:
            raise RuntimeError(f"Failed to download model '{model_id}': {e}") from e

    def _patch_config(self, model: Any, tokenizer: Any, language: str) -> None:
        """Applies critical fixes to GenerationConfig for fine-tuned models."""
        # Use Any to bypass Pylance strict checks on dynamic attributes
        gen_config: Any = model.generation_config

        if not getattr(gen_config, "no_timestamps_token_id", None):
            try:
                base = "openai/whisper-large-v2"
                if "large-v3" in getattr(model.config, "_name_or_path", "").lower():
                    base = "openai/whisper-large-v3"
                model.generation_config = GenerationConfig.from_pretrained(base)
                gen_config = model.generation_config
            except Exception:
                pass

        gen_config.language = language
        gen_config.task = "transcribe"
        gen_config.forced_decoder_ids = None

        if hasattr(tokenizer, "convert_tokens_to_ids"):
            notimestamp_id = tokenizer.convert_tokens_to_ids("<|notimestamps|>")
            if notimestamp_id is not None:
                gen_config.no_timestamps_token_id = notimestamp_id

    def _load_pipeline(self, language: str, exclude_models: list[str]) -> str:
        """Loads best available ASR model, skipping excluded ones. Returns model_id."""
        candidates = settings.whisper_model_map.get(language, [])
        if settings.fallback_model_id not in candidates:
            candidates.append(settings.fallback_model_id)

        valid_candidates = [m for m in candidates if m not in exclude_models]

        if not valid_candidates:
            raise RuntimeError(f"All model candidates failed for language '{language}'")

        for model_id in valid_candidates:
            if self._pipe and self._current_model_id == model_id:
                return model_id

            try:
                self._attempt_load_model(model_id, language)
                return model_id
            except Exception as e:
                print(f"[WARN] Failed to load '{model_id}': {e}")
                print("[INFO] Attempting next candidate...")
                self._clear_gpu()

        raise RuntimeError(f"All valid candidates failed for language '{language}'")

    def _attempt_load_model(self, model_id: str, language: str) -> None:
        """Helper to load a specific model."""
        self._clear_gpu()

        model_dir = self._resolve_model_path(model_id)

        tokenizer = AutoTokenizer.from_pretrained(model_dir)
        processor = AutoProcessor.from_pretrained(model_dir)

        model = AutoModelForSpeechSeq2Seq.from_pretrained(
            model_dir,
            torch_dtype=settings.torch_dtype,
            low_cpu_mem_usage=True,
            attn_implementation="sdpa",
        )
        model.to(settings.device)

        self._patch_config(model, tokenizer, language)

        self._pipe = pipeline(
            "automatic-speech-recognition",
            model=model,
            tokenizer=tokenizer,
            feature_extractor=processor.feature_extractor,
            chunk_length_s=settings.chunk_length_s,
            batch_size=settings.batch_size,
            torch_dtype=settings.torch_dtype,
            device=settings.device_index,
        )
        self._current_model_id = model_id
        print(f"[SUCCESS] Loaded Model: {model_id}")

    def _clear_gpu(self) -> None:
        """Clears GPU memory."""
        if self._pipe:
            del self._pipe
        self._pipe = None
        self._current_model_id = None
        gc.collect()
        torch.cuda.empty_cache()

    def _format_timestamp(self, seconds: float) -> str:
        """Formats seconds into SRT timestamp."""
        hours = int(seconds // 3600)
        minutes = int((seconds % 3600) // 60)
        secs = seconds % 60
        millis = int(round((secs - math.floor(secs)) * 1000))
        if millis == 1000:
            secs += 1
            millis = 0
        return f"{hours:02}:{minutes:02}:{int(secs):02},{millis:03}"

    def _write_srt(
        self, chunks: list[dict[str, Any]], path: Path, offset: float
    ) -> int:
        """Writes chunks to SRT. Returns number of lines written."""
        count = 0
        with open(path, "w", encoding="utf-8") as f:
            for idx, chunk in enumerate(chunks, start=1):
                text = chunk.get("text", "").strip()
                timestamp = chunk.get("timestamp")

                if not text or timestamp is None:
                    continue

                try:
                    start, end = timestamp
                except ValueError:
                    continue

                if start is None:
                    continue
                if end is None:
                    end = start + 2.0

                f.write(
                    f"{idx}\n"
                    f"{self._format_timestamp(start + offset)} --> "
                    f"{self._format_timestamp(end + offset)}\n"
                    f"{text}\n\n"
                )
                count += 1
        return count

    def transcribe(
        self,
        audio_path: Path,
        language: str | None = None,
        subtitle_path: Path | None = None,
        output_path: Path | None = None,
        start_time: float = 0.0,
        end_time: float | None = None,
    ) -> None:
        """Executes the transcription workflow with robustness loops."""
        if not audio_path.exists():
            print(f"[ERROR] Audio file not found: {audio_path}")
            return

        lang = language or settings.language
        out_srt = output_path or audio_path.with_suffix(".srt")

        # 1. Check existing subtitles
        if start_time == 0.0 and end_time is None:
            if self._find_existing_subtitles(audio_path, out_srt, subtitle_path, lang):
                return

        # 2. Prepare Audio
        proc_path = audio_path
        is_sliced = False
        if start_time > 0 or end_time is not None:
            proc_path = self._slice_audio(audio_path, start_time, end_time)
            is_sliced = True

        # 3. Robust Transcription Loop
        exclude_models: list[str] = []

        try:
            while True:
                try:
                    current_model = self._load_pipeline(lang, exclude_models)
                except RuntimeError as e:
                    print(f"[ERROR] Transcription failed: {e}")
                    break

                print(f"[INFO] Transcribing (Lang: {lang}) with {current_model}...")

                if self._pipe is None:
                    print("[ERROR] Pipeline is unexpectedly None.")
                    break

                try:
                    result = self._pipe(
                        str(proc_path),
                        return_timestamps=True,
                        generate_kwargs={"language": lang, "task": "transcribe"},
                    )
                    chunks = (
                        result.get("chunks", []) if isinstance(result, dict) else []
                    )

                    print(f"[DEBUG] Raw chunks generated: {len(chunks)}")

                    # Write and Validate
                    lines_written = self._write_srt(chunks, out_srt, offset=start_time)

                    if lines_written > 0:
                        print(
                            f"[SUCCESS] Saved {lines_written} subtitles to: {out_srt}"
                        )
                        break
                    else:
                        print(
                            f"[WARN] Model {current_model} produced empty/invalid SRT. "
                            "Retrying next model..."
                        )
                        exclude_models.append(current_model)
                        continue

                except Exception as e:
                    print(f"[ERROR] Inference error with {current_model}: {e}")
                    exclude_models.append(current_model)
                    continue

        finally:
            if is_sliced and proc_path.exists():
                try:
                    proc_path.unlink()
                except Exception:
                    pass


def main() -> None:
    """CLI Entry point."""
    if len(sys.argv) < 2:
        print(
            "Usage: python -m core.processing.transcriber "
            "<audio> [start] [end] [subtitle] [lang]"
        )
        sys.exit(1)

    args = sys.argv
    audio = Path(args[1]).resolve()
    start = float(args[2]) if len(args) > 2 else 0.0
    end = float(args[3]) if len(args) > 3 and args[3].lower() != "none" else None
    sub = (
        Path(args[4]).resolve() if len(args) > 4 and args[4].lower() != "none" else None
    )
    lang = args[5] if len(args) > 5 else None

    app = AudioTranscriber()
    app.transcribe(audio, lang, sub, output_path=None, start_time=start, end_time=end)


if __name__ == "__main__":
    main()
