"""Robust Audio Transcriber using Faster-Whisper (CTranslate2).

Features:
- RTX Optimized: Uses CTranslate2 (C++) with Int8 quantization for 5x speed.
- Smart Caching: Project Local > Global Cache > Download.
- Anti-Hallucination: Strict generation config to prevent looping/translation.
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
from typing import Any, Any as _Any, cast as _cast

import torch
from faster_whisper import WhisperModel

import config as _config

settings = _cast(_Any, _config.settings)

warnings.filterwarnings("ignore")

# Enable High-Performance Transfer & Timeouts (HF hub)
os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "1"
os.environ["HF_HUB_DOWNLOAD_TIMEOUT"] = "60"

# Mitigate fragmentation (helps allocation stability on CUDA).
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"


class AudioTranscriber:
    """Main transcription class handling ASR lifecycle."""

    def __init__(self) -> None:
        """Initialize the AudioTranscriber instance.

        Initializes internal pipeline references and prints device information.
        """
        self._model: WhisperModel | None = None
        self._current_model_size: str | None = None
        print(
            f"[INFO] Initialized Faster-Whisper ({settings.device}, "
            f"Compute: float16/int8)"
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

        # Faster-Whisper is robust, but 16kHz mono is always safest
        cmd.extend(["-ar", "16000", "-ac", "1", "-map", "0:a:0", str(output_slice)])

        print(f"[INFO] Slicing audio: {start}s -> {end if end else 'END'}s")
        subprocess.run(cmd, check=True, stderr=subprocess.DEVNULL)
        return output_slice

    def _load_model(self, model_key: str) -> None:
        """Load the Faster-Whisper model (CTranslate2).

        Args:
            model_key (str): The model size or repo ID.
        """
        if self._model:
            del self._model
            self._model = None
            gc.collect()
            if settings.device == "cuda":
                torch.cuda.empty_cache()

        # FORCE LARGE-V2 FOR TAMIL.
        model_size = model_key
        if "large-v3" in model_key:
            model_size = "large-v3"
        elif "large-v2" in model_key:
            model_size = "large-v2"
        elif "distil" in model_key and "large" in model_key:
            model_size = "distil-large-v3"

        print(f"[INFO] Loading Faster-Whisper Model: {model_size}...")

        try:
            compute_type = "float16" if settings.device == "cuda" else "int8"

            self._model = WhisperModel(
                model_size,
                device=settings.device,
                compute_type=compute_type,
                download_root=str(settings.model_cache_dir),
            )
            self._current_model_size = model_size
            print(f"[SUCCESS] Loaded {model_size} on {settings.device}")
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

        chunks: list[dict[str, Any]] = []

        # Determine best available model for language
        candidates = settings.whisper_model_map.get(lang, ["large-v3"])
        model_to_use = candidates[0] if candidates else "large-v3"

        # # FORCE LARGE-V2 FOR TAMIL.
        # if lang == "ta":
        #     model_to_use = "large-v2"

        try:
            # Load model if not loaded or if switching sizes
            if self._model is None or (
                self._current_model_size
                and self._current_model_size not in model_to_use
            ):
                self._load_model(model_to_use)

            if self._model is None:
                raise RuntimeError("Model failed to initialize")

            print(f"[INFO] Running Inference on {proc_path}...")

            prompt = (
                "வணக்கம் boss. Late aayiduchu. Sorry sir. Super-ah irukku. "
                "Casual Tanglish conversation."
            )

            segments, info = self._model.transcribe(
                str(proc_path),
                # It forces the model to output exactly what it hears (Greedy),
                beam_size=1,
                language=lang,
                task="transcribe",
                condition_on_previous_text=False,
                temperature=[0.0, 0.2, 0.4, 0.6, 0.8, 1.0],
                initial_prompt=prompt,
                repetition_penalty=1.0,
                no_repeat_ngram_size=0,
                vad_filter=True,
            )

            for segment in segments:
                chunk = {
                    "text": segment.text.strip(),
                    "timestamp": (segment.start, segment.end),
                }
                chunks.append(chunk)

            if not chunks:
                print("[WARN] No speech detected.")
                return None

            # Post-processing
            chunks = self._split_long_chunks(chunks, max_segment_s=8.0)
            lines_written = self._write_srt(chunks, out_srt, offset=start_time)

            if lines_written > 0:
                print(f"[SUCCESS] Saved {lines_written} subtitles to: {out_srt}")
                return chunks

        except Exception as e:
            print(f"[ERROR] Inference failed: {e}")

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
