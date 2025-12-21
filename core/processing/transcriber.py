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

import gc
import json
import os
import shutil
import subprocess
import sys
import warnings
from pathlib import Path
from tempfile import NamedTemporaryFile
from typing import Any

import ctranslate2.converters
import torch
from faster_whisper import BatchedInferencePipeline, WhisperModel
from huggingface_hub import snapshot_download

from config import settings
from core.processing.text_utils import parse_srt
from core.utils.logger import log
from core.utils.observe import observe

warnings.filterwarnings("ignore")

os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "0"
os.environ["HF_HUB_DOWNLOAD_TIMEOUT"] = "300"
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"


class AudioTranscriber:
    """Main transcription class handling ASR lifecycle."""

    def __init__(self) -> None:
        """Initialize the AudioTranscriber instance.

        Initializes internal pipeline references and prints device information.
        """
        self._model: WhisperModel | None = None
        self._batched_model: BatchedInferencePipeline | None = None
        self._current_model_size: str | None = None
        self.device = settings.device
        self.compute_type: str = "int8_float16" if self.device == "cuda" else "int8"

    def __enter__(self):
        """Called when entering the 'with' block."""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Called automatically when exiting the 'with' block."""
        self.unload_model()

    def unload_model(self) -> None:
        """Forcefully unload the Whisper model from VRAM."""
        log("[INFO] Unloading Whisper model to free VRAM...")

        # 1. Delete the CTranslate2 model objects
        if self._batched_model is not None:
            del self._batched_model
            self._batched_model = None

        if self._model is not None:
            del self._model
            self._model = None
            self._current_model_size = None

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
            raise RuntimeError("FFmpeg not found. Please install it to system PATH.")
        return cmd

    def _find_existing_subtitles(
        self,
        input_path: Path,
        output_path: Path,
        user_sub_path: Path | None,
        language: str,
    ) -> bool:
        """Searches for subtitles: user-provided, sidecar files, or embedded streams.

        Args:
            input_path (Path): Video/audio input file.
            output_path (Path): Target subtitle output path.
            user_sub_path (Path | None): Optional user override subtitle file.
            language (str): Language specifier (e.g., "en", "ta").

        Returns:
            bool: True if subtitles were found/extracted; False otherwise.
        """
        log("[INFO] Probing for existing subtitles...")

        # 1. Check for explicit user provided subtitle path
        if user_sub_path and user_sub_path.exists():
            log(f"[SUCCESS] Using provided subtitle: {user_sub_path}")
            shutil.copy(user_sub_path, output_path)
            return True

        # 2. Check for sidecar files (video.srt, video.lang.srt)
        # We check input_path directory for files with same stem
        for sidecar in [
            input_path.with_suffix(f".{language}.srt"),
            input_path.with_suffix(".srt"),
        ]:
            if sidecar.exists() and sidecar != output_path:
                log(f"[SUCCESS] Found sidecar: {sidecar}")
                shutil.copy(sidecar, output_path)
                return True

        # 3. Check for embedded subtitles using ffmpeg
        cmd = [
            self._get_ffmpeg_cmd(),
            "-y",
            "-v",
            "error",
            "-i",
            str(input_path),
            "-map",
            "0:s:0",  # Map first subtitle stream
            str(output_path),
        ]
        try:
            subprocess.run(
                cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL
            )
            if output_path.exists() and output_path.stat().st_size > 0:
                log("[SUCCESS] Extracted embedded subtitles.")
                return True
        except subprocess.CalledProcessError:
            pass

        log("[INFO] No existing subtitles found. Proceeding to ASR.")
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

        log(f"[INFO] Slicing audio: {start}s -> {end if end else 'END'}s")
        subprocess.run(cmd, check=True, stderr=subprocess.DEVNULL)
        return output_slice

    def _convert_and_cache_model(self, model_id: str) -> str:
        """Downloads/converts models using HF transfer and CTranslate2."""
        sanitized_name = model_id.replace("/", "_")
        ct2_output_dir = settings.model_cache_dir / "converted_models" / sanitized_name
        raw_model_dir = settings.model_cache_dir / "raw_models" / sanitized_name

        if (ct2_output_dir / "config.json").exists():
            try:
                with open(ct2_output_dir / "config.json", "r", encoding="utf-8") as f:
                    conf = json.load(f)
                    if "architectures" in conf or "_name_or_path" in conf:
                        log(f"[WARN] Corrupted config in {ct2_output_dir}. Purging.")
                        shutil.rmtree(ct2_output_dir)
            except Exception:
                shutil.rmtree(ct2_output_dir, ignore_errors=True)

        if (ct2_output_dir / "model.bin").exists():
            needed_files = ["tokenizer.json", "vocab.json"]
            if "large-v3" in model_id:
                needed_files.append("preprocessor_config.json")

            for fname in needed_files:
                if not (ct2_output_dir / fname).exists():
                    if (raw_model_dir / fname).exists():
                        shutil.copy(raw_model_dir / fname, ct2_output_dir / fname)
            return str(ct2_output_dir)

        log(f"[INFO] Model {model_id} needs conversion.")
        log(f"[INFO] High-Speed Download to {raw_model_dir}...")

        try:
            snapshot_download(
                repo_id=model_id,
                local_dir=str(raw_model_dir),
                local_dir_use_symlinks=False,
                resume_download=True,
                max_workers=8,
                token=settings.hf_token,
                ignore_patterns=["*.msgpack", "*.h5", "*.tflite", "*.ot"],
            )
            log("[SUCCESS] Download complete.")

            # Check if already CTranslate2 format
            if (raw_model_dir / "model.bin").exists() and (raw_model_dir / "config.json").exists():
                 log(f"[INFO] Model {model_id} appears to be already converted. Skipping conversion step.")
                 shutil.copytree(raw_model_dir, ct2_output_dir, dirs_exist_ok=True)
                 return str(ct2_output_dir)

            log(f"[INFO] Converting to CTranslate2 at {ct2_output_dir}...")

            converter = ctranslate2.converters.TransformersConverter(
                str(raw_model_dir), load_as_float16=True
            )
            converter.convert(str(ct2_output_dir), quantization="float16", force=True)

            for file_name in [
                "preprocessor_config.json",
                "tokenizer.json",
                "vocab.json",
            ]:
                src = raw_model_dir / file_name
                if src.exists():
                    shutil.copy(src, ct2_output_dir / file_name)

            log("[SUCCESS] Model converted and patched successfully.")
            return str(ct2_output_dir)

        except Exception as e:
            log(f"[ERROR] Model download/conversion failed: {e}")
            raise RuntimeError(f"Could not prepare {model_id}.") from e

    def _load_model(self, model_key: str) -> None:
        """Loads the Faster-Whisper model."""
        if self._model:
            del self._model
            del self._batched_model
            self._model = None
            self._batched_model = None
            gc.collect()
            if self.device == "cuda" and torch.cuda.is_available():
                torch.cuda.empty_cache()

        # Log only when loading weights
        log(
            f"[INFO] Loading Faster-Whisper model '{model_key}' "
            f"({self.device}, Compute: {self.compute_type})..."
        )

        final_model_path = self._convert_and_cache_model(model_key)
        try:
            self._model = WhisperModel(
                final_model_path,
                device=self.device,
                compute_type=self.compute_type,
                download_root=str(settings.model_cache_dir),
            )
            self._batched_model = BatchedInferencePipeline(model=self._model)
            self._current_model_size = model_key
            log(f"[SUCCESS] Loaded {model_key} on {self.device}")
        except Exception as e:
            raise RuntimeError(f"Failed to load Faster-Whisper: {e}") from e

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
                out.append({"text": sub_text, "timestamp": (sub_start, sub_end)})

            if ptr < len(words) and out:
                out[-1]["text"] = out[-1]["text"] + " " + " ".join(words[ptr:])

        return out

    @observe("transcription")
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
            list[dict[str, Any]] | None: list of ASR chunks, or None if subtitle
                fallback was used.
        """
        if not audio_path.exists():
            log(f"[ERROR] Input file not found: {audio_path}")
            return None

        lang = language or settings.language
        out_srt = output_path or audio_path.with_suffix(".srt")

        # Existing Sidecar / Embedded Subtitles
        if start_time == 0.0 and end_time is None:
            if self._find_existing_subtitles(audio_path, out_srt, subtitle_path, lang):
                log(f"[INFO] Parsing existing subtitles from {out_srt}...")
                parsed_segments = parse_srt(out_srt)
                log(
                    f"[SUCCESS] Loaded {len(parsed_segments)}"
                    " segments from existing file."
                )
                return parsed_segments

        # Run Whisper (If no subs found)
        is_sliced = True
        proc_path = self._slice_audio(audio_path, start_time, end_time)
        chunks: list[dict[str, Any]] = []

        candidates = settings.whisper_model_map.get(lang, [settings.fallback_model_id])
        model_to_use = candidates[0] if candidates else settings.fallback_model_id

        try:
            if self._batched_model is None or (
                self._current_model_size != model_to_use
            ):
                self._load_model(model_to_use)

            if self._batched_model is None:
                raise RuntimeError("Model failed to initialize")

            log(f"[INFO] Running Inference on {proc_path} with {model_to_use}.")

            segments, info = self._batched_model.transcribe(
                str(proc_path),
                batch_size=settings.batch_size,
                language=lang,
                task="transcribe",
                beam_size=3,
                condition_on_previous_text=False,
                initial_prompt=None,
                repetition_penalty=1.2,
                word_timestamps=True,
                vad_filter=True,
                vad_parameters={
                    "min_speech_duration_ms": 250,
                    "min_silence_duration_ms": 500,
                    "speech_pad_ms": 500,
                    "threshold": 0.4,
                },
                no_speech_threshold=0.6,
                log_prob_threshold=-1.0,
            )

            for segment in segments:
                if segment.words:
                    current_words = []
                    seg_start = segment.words[0].start

                    for word in segment.words:
                        current_words.append(word.word)
                        if (word.end - seg_start > 6.0) or word.word.strip().endswith(
                            ("?", ".", "!")
                        ):
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

            log(f"[SUCCESS] Transcription complete. Prob: {info.language_probability}")

            lines_written = self._write_srt(chunks, out_srt, offset=start_time)
            if lines_written > 0:
                log(f"[SUCCESS] Saved {lines_written} subtitles to: {out_srt}")
                return chunks

        except Exception as e:
            log(f"[ERROR] Inference failed: {e}")
            raise
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
    lang = args[5] if len(args) > 5 else "ta"

    AudioTranscriber().transcribe(audio, lang, sub, None, start, end)


if __name__ == "__main__":
    main()
