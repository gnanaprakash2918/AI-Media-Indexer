"""Robust Audio Transcriber with Subtitle Fallback and RTX Optimization.

Features:
- Fast Downloads: Uses hf_transfer (Rust) and resumable downloads.
- Smart Caching: Project Local > Global Cache > Download.
- Self-Healing: Retries download if cache is corrupted.
- Anti-Hallucination: Disables previous text conditioning.
- Pipeline Ready: Returns in-memory chunks for Vector DB ingestion.
"""

import gc
import os
import shutil
import subprocess
import sys
import warnings
from pathlib import Path
from tempfile import NamedTemporaryFile
from typing import Any, cast

import torch
from huggingface_hub import snapshot_download
from transformers import (
    AutoModelForSpeechSeq2Seq,
    AutoProcessor,
    AutoTokenizer,
)
from transformers.generation.configuration_utils import GenerationConfig
from transformers.modeling_utils import PreTrainedModel
from transformers.pipelines import pipeline
from transformers.pipelines.automatic_speech_recognition import (
    AutomaticSpeechRecognitionPipeline,
)
from transformers.pipelines.base import Pipeline
from transformers.utils import logging as hf_logging

from config import settings

# Cleanup console output
warnings.filterwarnings("ignore")
hf_logging.set_verbosity_error()

# Enable High-Performance Transfer & Timeouts
os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "1"
os.environ["HF_HUB_DOWNLOAD_TIMEOUT"] = "60"


class AudioTranscriber:
    """Main transcription class handling ASR lifecycle."""

    def __init__(self) -> None:
        """Initialize the transcriber."""
        self._pipe: Pipeline | None = None
        self._current_model_id: str | None = None
        print(
            f"[INFO] Initialized Transcriber ({settings.device}, "
            f"{settings.torch_dtype})"
        )

    def _get_ffmpeg_cmd(self) -> str:
        """Locate FFmpeg executable."""
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
        """Probes for existing subtitles in input, sidecars, or embedded streams."""
        print("[INFO] Probing for existing subtitles...")

        # 1. User Provided
        if user_sub_path and user_sub_path.exists():
            print(f"[SUCCESS] Using provided subtitle: {user_sub_path}")
            shutil.copy(user_sub_path, output_path)
            return True

        for sidecar in [
            input_path.with_suffix(f".{language}.srt"),
            input_path.with_suffix(".srt"),
        ]:
            if sidecar.exists() and sidecar != output_path:
                print(f"[SUCCESS] Found sidecar: {sidecar}")
                shutil.copy(sidecar, output_path)
                return True

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
        """Slice audio safely using FFmpeg into a temp file."""
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

    def _resolve_model_path(self, model_id: str, force_download: bool = False) -> str:
        """Resolve model path: Project Cache > Global Cache > Download."""
        kwargs = {
            "repo_id": model_id,
            "token": settings.hf_token,
            "ignore_patterns": ["*.msgpack", "*.h5", "*.ot"],
        }

        if not force_download:
            try:
                print(f"[INFO] Found '{model_id}' in project cache.")
                return str(
                    snapshot_download(
                        **kwargs,
                        cache_dir=settings.model_cache_dir,
                        local_files_only=True,
                    )
                )
            except (OSError, ValueError):
                pass
            try:
                print(f"[INFO] Found '{model_id}' in global cache.")
                return str(
                    snapshot_download(**kwargs, cache_dir=None, local_files_only=True)
                )
            except (OSError, ValueError):
                pass

        print(f"[INFO] Downloading '{model_id}'...")
        try:
            return str(
                snapshot_download(
                    **kwargs,
                    cache_dir=settings.model_cache_dir,
                    local_files_only=False,
                    resume_download=True,
                    max_workers=8,
                )
            )
        except Exception as e:
            raise RuntimeError(f"Download failed: {e}") from e

    def _patch_config(self, model: Any, tokenizer: Any, language: str) -> None:
        """Apply critical fixes to GenerationConfig for fine-tuned models."""
        gen_config = model.generation_config
        if not getattr(gen_config, "no_timestamps_token_id", None):
            try:
                model.generation_config = GenerationConfig.from_pretrained(
                    "openai/whisper-large-v3"
                )
                gen_config = model.generation_config
            except Exception:
                pass

        # Type ignores added because Pylance struggles with dynamic config attributes
        gen_config.language = language  # type: ignore
        gen_config.task = "transcribe"  # type: ignore
        gen_config.forced_decoder_ids = None  # type: ignore
        if hasattr(gen_config, "condition_on_previous_text"):
            gen_config.condition_on_previous_text = False  # type: ignore

        if hasattr(tokenizer, "convert_tokens_to_ids"):
            tid = tokenizer.convert_tokens_to_ids("<|notimestamps|>")
            if tid is not None:
                gen_config.no_timestamps_token_id = tid  # type: ignore

    def _attempt_load_model(
        self, model_id: str, language: str, force_download: bool
    ) -> None:
        """Helper to download and load a specific model."""
        self._clear_gpu()
        model_dir = self._resolve_model_path(model_id, force_download)

        try:
            tokenizer = AutoTokenizer.from_pretrained(model_dir)
            processor = AutoProcessor.from_pretrained(model_dir)
            model = AutoModelForSpeechSeq2Seq.from_pretrained(
                model_dir,
                torch_dtype=settings.torch_dtype,
                low_cpu_mem_usage=True,
                attn_implementation="sdpa",
            )
            model.to(settings.device)

            if settings.device == "cuda":
                try:
                    model = torch.compile(  # type: ignore
                        model, mode="reduce-overhead", fullgraph=True
                    )
                except Exception:
                    pass

            self._patch_config(model, tokenizer, language)
            safe_model = cast(PreTrainedModel, model)

            self._pipe = pipeline(
                "automatic-speech-recognition",
                model=safe_model,
                tokenizer=tokenizer,
                feature_extractor=processor.feature_extractor,
                chunk_length_s=settings.chunk_length_s,
                batch_size=settings.batch_size,
                torch_dtype=settings.torch_dtype,
                device=settings.device_index,
            )
            self._current_model_id = model_id
            print(f"[SUCCESS] Loaded: {model_id}")
        except Exception as e:
            raise RuntimeError(f"Init failed: {e}") from e

    def _load_pipeline(self, language: str, exclude_models: list[str]) -> str:
        """Load best available ASR model, skipping excluded ones."""
        candidates = settings.whisper_model_map.get(language, [])
        if settings.fallback_model_id not in candidates:
            candidates.append(settings.fallback_model_id)

        valid = [m for m in candidates if m not in exclude_models]
        if not valid:
            raise RuntimeError(f"No models available for {language}")

        for model_id in valid:
            if self._pipe and self._current_model_id == model_id:
                return model_id
            try:
                self._attempt_load_model(model_id, language, False)
                return model_id
            except Exception:
                print(f"[WARN] Load failed '{model_id}', retrying download...")
                try:
                    self._attempt_load_model(model_id, language, True)
                    return model_id
                except Exception:
                    self._clear_gpu()
                    continue
        raise RuntimeError("All models failed.")

    def _clear_gpu(self) -> None:
        """Clear GPU memory."""
        if self._pipe:
            del self._pipe
        self._pipe = None
        self._current_model_id = None
        gc.collect()
        torch.cuda.empty_cache()

    def _format_timestamp(self, seconds: float) -> str:
        """Format seconds into SRT timestamp."""
        hours, remainder = divmod(seconds, 3600)
        minutes, secs = divmod(remainder, 60)
        millis = int(round((secs - int(secs)) * 1000))
        return f"{int(hours):02}:{int(minutes):02}:{int(secs):02},{millis:03}"

    def _write_srt(
        self, chunks: list[dict[str, Any]], path: Path, offset: float
    ) -> int:
        """Write chunks to SRT with Hallucination Filtering."""
        count = 0
        last_text = ""
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

                if (end - start) < 0.2:
                    continue
                if text == last_text:
                    continue

                f.write(
                    f"{count + 1}\n"
                    f"{self._format_timestamp(start + offset)} --> "
                    f"{self._format_timestamp(end + offset)}\n"
                    f"{text}\n\n"
                )
                count += 1
                last_text = text
        return count

    def transcribe(
        self,
        audio_path: Path,
        language: str | None = None,
        subtitle_path: Path | None = None,
        output_path: Path | None = None,
        start_time: float = 0.0,
        end_time: float | None = None,
    ) -> list[dict[str, Any]] | None:
        """Execute the transcription workflow with robustness loops."""
        if not audio_path.exists():
            return None

        lang = language or settings.language
        out_srt = output_path or audio_path.with_suffix(".srt")

        if start_time == 0.0 and end_time is None:
            if self._find_existing_subtitles(audio_path, out_srt, subtitle_path, lang):
                return None

        proc_path = audio_path
        is_sliced = False
        if start_time > 0 or end_time is not None:
            proc_path = self._slice_audio(audio_path, start_time, end_time)
            is_sliced = True

        exclude: list[str] = []
        try:
            while True:
                try:
                    current_model = self._load_pipeline(lang, exclude)
                except RuntimeError:
                    break

                if self._pipe is None:
                    break

                try:
                    asr_pipeline: Any = self._pipe

                    prompt = (
                        "This is a casual vlog. Vanakkam, hello guys, welcome back. "
                        "Namba channel-la paarkalam. Super-ah iruku."
                    )
                    prompt_ids = None

                    typed_pipe = cast(AutomaticSpeechRecognitionPipeline, self._pipe)
                    if typed_pipe.tokenizer:
                        try:
                            prompt_ids = typed_pipe.tokenizer.get_prompt_ids(  # type: ignore
                                prompt, return_tensors="pt"
                            ).to(settings.device)
                        except AttributeError:
                            pass

                    gen_kwargs = {
                        "language": lang,
                        "task": "transcribe",
                        "temperature": 0.2,
                        "repetition_penalty": 1.2,
                        "no_repeat_ngram_size": 3,
                        "prompt_ids": prompt_ids,
                    }

                    result = asr_pipeline(
                        str(proc_path),
                        return_timestamps=True,
                        generate_kwargs=gen_kwargs,
                    )
                    chunks = (
                        result.get("chunks", []) if isinstance(result, dict) else []
                    )

                    if self._write_srt(chunks, out_srt, offset=start_time) > 0:
                        print(f"[SUCCESS] Saved subtitles: {out_srt}")
                        return chunks

                    exclude.append(current_model)
                except Exception as e:
                    print(f"[ERROR] {current_model}: {e}")
                    exclude.append(current_model)

        finally:
            if is_sliced and proc_path.exists():
                try:
                    proc_path.unlink()
                except Exception:
                    pass
        return None


def main() -> None:
    """CLI Entry point."""
    if len(sys.argv) < 2:
        sys.exit(1)
    args = sys.argv
    audio = Path(args[1]).resolve()
    start = float(args[2]) if len(args) > 2 else 0.0
    end = float(args[3]) if len(args) > 3 and args[3].lower() != "none" else None
    sub = (
        Path(args[4]).resolve() if len(args) > 4 and args[4].lower() != "none" else None
    )
    lang = args[5] if len(args) > 5 else None

    AudioTranscriber().transcribe(audio, lang, sub, None, start, end)


if __name__ == "__main__":
    main()
