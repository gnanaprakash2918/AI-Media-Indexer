"""Robust Audio Transcriber with Subtitle Fallback and RTX Optimization.

Features:
- Fast Downloads: Uses hf_transfer (Rust) and resumable downloads.
- Smart Caching: Project Local > Global Cache > Download.
- Self-Healing: Retries download if cache is corrupted.
- Anti-Hallucination: Disables previous text conditioning.
- Pipeline Ready: Returns in-memory chunks for Vector DB ingestion.
"""

# Ensure environment hints are set before torch is imported.
import os

# Mitigate fragmentation (helps allocation stability on CUDA).
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
# If you want to force CPU for quick debugging, uncomment:
# os.environ["CUDA_VISIBLE_DEVICES"] = ""

import gc
import shutil
import subprocess
import sys
import warnings
from pathlib import Path
from tempfile import NamedTemporaryFile
from typing import Any, Any as _Any, cast, cast as _cast

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

import config as _config

settings = _cast(_Any, _config.settings)

warnings.filterwarnings("ignore")
hf_logging.set_verbosity_error()

# Enable High-Performance Transfer & Timeouts (HF hub)
os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "1"
os.environ["HF_HUB_DOWNLOAD_TIMEOUT"] = "60"


class AudioTranscriber:
    """Main transcription class handling ASR lifecycle."""

    def __init__(self) -> None:
        """Initialize the AudioTranscriber instance.

        Initializes internal pipeline references and prints device information.
        """
        self._pipe: Pipeline | None = None
        self._current_model_id: str | None = None
        print(
            f"[INFO] Initialized Transcriber ({settings.device}, "
            f"{settings.torch_dtype})"
        )

    def _get_ffmpeg_cmd(self) -> str:
        """Locate the ffmpeg executable in system PATH.

        Returns:
            str: Full path to the ffmpeg executable.

        Raises:
            RuntimeError: If ffmpeg is not installed or not found in PATH.
        """
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
        """Search for subtitles: user-provided, sidecar files, or embedded streams.

        Args:
            input_path (Path): Video/audio input file.
            output_path (Path): Target subtitle output path.
            user_sub_path (Path | None): Optional user override subtitle file.
            language (str): Language specifier (e.g., "en", "ta").

        Returns:
            bool: True if subtitles were found/extracted; False otherwise.
        """
        print("[INFO] Probing for existing subtitles...")

        # 1. User Provided
        if user_sub_path and user_sub_path.exists():
            print(f"[SUCCESS] Using provided subtitle: {user_sub_path}")
            shutil.copy(user_sub_path, output_path)
            return True

        # 2. Sidecar files
        for sidecar in [
            input_path.with_suffix(f".{language}.srt"),
            input_path.with_suffix(".srt"),
        ]:
            if sidecar.exists() and sidecar != output_path:
                print(f"[SUCCESS] Found sidecar: {sidecar}")
                shutil.copy(sidecar, output_path)
                return True

        # 3. Embedded subtitles extraction
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
        """Slice audio using ffmpeg into a temporary WAV file.

        Args:
            input_path (Path): Original audio file.
            start (float): Start time in seconds.
            end (float | None): End time in seconds or None for full remaining audio.

        Returns:
            Path: Path to the temporary sliced audio file.
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
        cmd.extend(["-ar", "16000", "-ac", "1", "-map", "0:a:0", str(output_slice)])

        print(f"[INFO] Slicing audio: {start}s -> {end if end else 'END'}s")
        subprocess.run(cmd, check=True, stderr=subprocess.DEVNULL)
        return output_slice

    def _resolve_model_path(self, model_id: str, force_download: bool = False) -> str:
        """Resolve a model path from project cache, global cache, or download.

        Args:
            model_id (str): HuggingFace model ID.
            force_download (bool): If True, forces re-download.

        Returns:
            str: Filesystem path to the resolved model.

        Raises:
            RuntimeError: If the model fails to download.
        """
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
        """Apply Whisper-generation config patches for fine-tuned models.

        Args:
            model (Any): Loaded ASR model.
            tokenizer (Any): Model tokenizer.
            language (str): Target language code.

        Notes:
            This modifies dynamic attributes on the model's GenerationConfig.
        """
        gen_config = model.generation_config
        if not getattr(gen_config, "no_timestamps_token_id", None):
            try:
                model.generation_config = GenerationConfig.from_pretrained(
                    "openai/whisper-large-v3"
                )
                gen_config = model.generation_config
            except Exception:
                pass

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
        """Load a model from cache or download, then build the ASR pipeline.

        Args:
            model_id (str): Model identifier.
            language (str): Target transcription language.
            force_download (bool): Whether to force model download.

        Raises:
            RuntimeError: If model initialization fails.
        """
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
                    try:
                        total_mem = torch.cuda.get_device_properties(0).total_memory
                    except Exception:
                        total_mem = 0
                    if total_mem >= 16 * 1024**3:
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
        """Load the best available model pipeline, skipping excluded ones.

        Args:
            language (str): Target transcription language.
            exclude_models (list[str]): Models previously attempted or failed.

        Returns:
            str: The model ID successfully loaded.

        Raises:
            RuntimeError: If all model candidates fail to load.
        """
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
                # Retry by forcing a download if the first attempt fails.
                print(f"[WARN] Load failed '{model_id}', retrying download...")
                try:
                    self._attempt_load_model(model_id, language, True)
                    return model_id
                except Exception:
                    self._clear_gpu()
                    continue
        raise RuntimeError("All models failed.")

    def _clear_gpu(self) -> None:
        """Release GPU memory and clear pipeline references."""
        if self._pipe:
            del self._pipe
        self._pipe = None
        self._current_model_id = None
        gc.collect()
        torch.cuda.empty_cache()

    def _format_timestamp(self, seconds: float) -> str:
        """Convert seconds into SRT timestamp format.

        Args:
            seconds (float): Seconds value.

        Returns:
            str: Formatted timestamp ("HH:MM:SS,mmm").
        """
        hours, remainder = divmod(seconds, 3600)
        minutes, secs = divmod(remainder, 60)
        millis = int(round((secs - int(secs)) * 1000))
        return f"{int(hours):02}:{int(minutes):02}:{int(secs):02},{millis:03}"

    def _write_srt(
        self, chunks: list[dict[str, Any]], path: Path, offset: float
    ) -> int:
        """Write ASR output chunks to an SRT subtitle file.

        Args:
            chunks (list[dict]): List of timestamped transcribed segments.
            path (Path): Output SRT file path.
            offset (float): Offset to apply to timestamps.

        Returns:
            int: Number of subtitle entries written.
        """
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

                # Maintain previous behavior: skip if start missing
                if start is None:
                    continue
                # Provide default end when missing
                if end is None:
                    end = start + 2.0

                # Make types explicit for the type-checker and for arithmetic
                start = float(start)
                end = float(end)

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

    def _split_long_chunks(
        self, chunks: list[dict[str, Any]], max_segment_s: float = 8.0
    ) -> list[dict[str, Any]]:
        """Split chunks longer than `max_segment_s` into smaller subchunks.

        The text is split approximately proportionally across time bins by word
        count. This is a heuristic to avoid extremely long subtitle segments.

        Args:
            chunks (list[dict]): Original chunks from the pipeline.
            max_segment_s (float): Maximum allowed duration per SRT entry.

        Returns:
            list[dict]: New list of chunks where long entries are subdivided.
        """
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

            # number of splits (ceil)
            n = int((duration + max_segment_s - 1e-9) // max_segment_s) + 1
            n = max(1, n)
            words = text.split()
            if not words:
                # no words present, fallback to original chunk
                out.append(ch)
                continue

            # compute per-split approximate word counts
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
                out.append({"text": sub_text, "timestamp": (sub_start, sub_end)})

            # attach leftovers to last
            if ptr < len(words) and out:
                out[-1]["text"] = out[-1]["text"] + " " + " ".join(words[ptr:])

        return out

    def _rebuild_chunks_from_word_items(self, result: Any) -> list[dict[str, Any]]:
        """Rebuild chunks using per-word/token items when available.

        This tries to preserve tokens/words exactly as produced by the model
        instead of relying on the normalized `chunk['text']`. This helps keep
        English words and token-level outputs unchanged.

        Args:
            result (Any): The raw result returned by the ASR pipeline. Usually a
                dict.

        Returns:
            list[dict]: Normalized chunks with `text` and `timestamp` keys.
        """
        rebuilt: list[dict[str, Any]] = []

        chunks = []
        if isinstance(result, dict) and "chunks" in result:
            chunks = result.get("chunks", []) or []
        elif isinstance(result, list):
            chunks = result
        else:
            return rebuilt

        for ch in chunks:
            text = ""
            timestamp = ch.get("timestamp")

            words_field = (
                ch.get("words") or ch.get("word_timestamps") or ch.get("tokens")
            )

            if (
                words_field
                and isinstance(words_field, (list, tuple))
                and len(words_field) > 0
            ):
                pieces: list[str] = []
                w_start = None
                w_end = None
                for item in words_field:
                    if isinstance(item, dict):
                        token_text = (
                            item.get("text")
                            or item.get("word")
                            or item.get("token")
                            or ""
                        )
                        if token_text is None:
                            token_text = ""
                        pieces.append(token_text)

                        istart = item.get("start")
                        iend = item.get("end")

                        if w_start is None and istart is not None:
                            w_start = float(istart)
                        if iend is not None:
                            w_end = float(iend)
                    else:
                        pieces.append(str(item))

                text = " ".join([p for p in pieces if p is not None and p != ""])

                if (not timestamp or timestamp is None) and (
                    w_start is not None and w_end is not None
                ):
                    timestamp = (w_start, w_end)
            else:
                text = (ch.get("text") or "").strip()

            if not timestamp:
                continue

            try:
                s, e = timestamp
            except Exception:
                continue

            if s is None:
                continue
            if e is None:
                e = s + 2.0

            s = float(s)
            e = float(e)

            rebuilt.append({"text": text, "timestamp": (s, e)})

        return rebuilt

    def transcribe(
        self,
        audio_path: Path,
        language: str | None = None,
        subtitle_path: Path | None = None,
        output_path: Path | None = None,
        start_time: float = 0.0,
        end_time: float | None = None,
    ) -> list[dict[str, Any]] | None:
        """Main entry point: Transcribe audio and generate SRT subtitles.

        Args:
            audio_path (Path): Path to the input audio/video file.
            language (str | None): Target language code or fallback setting.
            subtitle_path (Path | None): Optional external subtitle to override.
            output_path (Path | None): Output SRT destination (auto if None).
            start_time (float): Start offset in seconds.
            end_time (float | None): End offset in seconds.

        Returns:
            list[dict[str, Any]] | None: List of ASR chunks, or None if subtitle
                fallback was used.
        """
        if not audio_path.exists():
            return None

        lang = language or settings.language
        out_srt = output_path or audio_path.with_suffix(".srt")

        # Existing subtitle detection
        if start_time == 0.0 and end_time is None:
            if self._find_existing_subtitles(audio_path, out_srt, subtitle_path, lang):
                return None

        # Slice audio if required
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
                            )
                            if prompt_ids is not None:
                                prompt_ids = prompt_ids.to(settings.device)
                        except AttributeError:
                            pass

                    gen_kwargs: dict[str, Any] = {
                        "task": "transcribe",
                        "temperature": 0.2,
                        "repetition_penalty": 1.2,
                        "no_repeat_ngram_size": 3,
                        "prompt_ids": prompt_ids,
                    }

                    use_word_timestamps = False
                    if self._current_model_id:
                        mid = self._current_model_id.lower()
                        if "large" not in mid:
                            use_word_timestamps = True

                    try:
                        if use_word_timestamps:
                            result = asr_pipeline(
                                str(proc_path),
                                return_timestamps="word",
                                generate_kwargs=gen_kwargs,
                            )
                        else:
                            result = asr_pipeline(
                                str(proc_path),
                                return_timestamps=True,
                                generate_kwargs=gen_kwargs,
                            )
                    except TypeError:
                        result = asr_pipeline(
                            str(proc_path),
                            return_timestamps=True,
                            generate_kwargs=gen_kwargs,
                        )

                    chunks = self._rebuild_chunks_from_word_items(result)

                    if not chunks:
                        if isinstance(result, dict):
                            chunks = result.get("chunks", []) or []
                        else:
                            chunks = []

                    # Split long chunks into shorter ones (heuristic)
                    chunks = self._split_long_chunks(chunks, max_segment_s=8.0)

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
    """CLI entry point to run the AudioTranscriber via command-line args.

    Usage:
        python script.py <audio> [start] [end] [subtitle] [language]
    """
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
