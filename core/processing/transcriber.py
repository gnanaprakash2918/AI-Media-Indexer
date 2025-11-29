"""Robust Audio Transcriber with Subtitle Fallback and HF Patching.

Prioritizes extracting existing subtitles. If none exist, performs ASR
using Hugging Face Transformers, applying specific patches for fine-tuned
models (e.g., Vasista) to prevent timestamp errors.
"""

import shutil
import subprocess
import sys
import tempfile
import warnings
from pathlib import Path

from transformers import (
    AutoModelForSpeechSeq2Seq,
    AutoProcessor,
    AutoTokenizer,
)
from transformers.pipelines import pipeline
from transformers.utils import logging as hf_logging

from config import settings

# Suppress noise
warnings.filterwarnings("ignore")
hf_logging.set_verbosity_error()


class AudioTranscriber:
    """Main transcription class handling ASR lifecycle and logic."""

    def __init__(self) -> None:
        """Initialize the Transcriber and load models lazily."""
        self._pipe = None
        self.device = settings.device
        self.dtype = settings.torch_dtype
        print(
            "Info: Initialized Transcriber "
            f"(Device: {self.device}, Dtype: {self.dtype})"
        )

    def _get_ffmpeg_cmd(self) -> str:
        """Locate FFmpeg executable."""
        cmd = shutil.which("ffmpeg")
        if not cmd:
            raise RuntimeError("FFmpeg not found. Please install it to system PATH.")
        return cmd

    def _extract_existing_subtitles(self, input_path: Path, output_path: Path) -> bool:
        """Try to extract embedded subtitles using FFmpeg."""
        print(f"Info: Probing for embedded subtitles in {input_path.name}...")

        # 1. Check for sidecar file first (e.g., movie.ta.srt)
        for ext in [f".{settings.language}.srt", ".srt"]:
            sidecar = input_path.with_suffix(ext)
            if sidecar.exists() and sidecar != output_path:
                print(f"Success: Found sidecar subtitle file: {sidecar}")
                shutil.copy(sidecar, output_path)
                return True

        # 2. Attempt FFmpeg extraction of internal stream (map 0:s:0)
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
                print("Success: Extracted embedded subtitles.")
                return True
        except subprocess.CalledProcessError:
            pass

        return False

    def _slice_audio(self, input_path: Path, start: float, end: float | None) -> Path:
        """Create a temporary audio slice using FFmpeg."""
        temp_dir = Path(tempfile.gettempdir())
        output_slice = temp_dir / f"slice_{input_path.stem}.wav"

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
        if end is not None:
            cmd.extend(["-to", str(end)])

        cmd.extend(["-ar", "16000", "-ac", "1", "-map", "0:a:0", str(output_slice)])

        print(f"Info: Slicing audio {start}s -> {end if end else 'END'}s...")
        subprocess.run(cmd, check=True)
        return output_slice

    def _patch_model_config(self, model, tokenizer) -> None:
        """Fix missing token IDs in generation config for fine-tuned models."""
        # Fix 1: Ensure no_timestamps_token_id is set (Critical for Vasista)
        if not getattr(model.generation_config, "no_timestamps_token_id", None):
            notimestamp_token = "<|notimestamps|>"
            token_id = tokenizer.convert_tokens_to_ids(notimestamp_token)
            if token_id is not None:
                model.generation_config.no_timestamps_token_id = token_id
                print(f"Debug: Patched 'no_timestamps_token_id' to {token_id}")

        # Fix 2: Clear forced_decoder_ids to allow language detection/setting
        if model.generation_config.forced_decoder_ids is not None:
            model.generation_config.forced_decoder_ids = None

        # Fix 3: Explicitly set language/task
        model.generation_config.language = settings.language
        model.generation_config.task = "transcribe"
        model.generation_config.is_multilingual = True

    def _load_pipeline(self) -> None:
        """Load the model pipeline only when needed."""
        if self._pipe is not None:
            return

        print(f"Info: Loading model '{settings.model_id}'...")
        try:
            tokenizer = AutoTokenizer.from_pretrained(
                settings.model_id, token=settings.hf_token
            )
            processor = AutoProcessor.from_pretrained(
                settings.model_id, token=settings.hf_token
            )
            model = AutoModelForSpeechSeq2Seq.from_pretrained(
                settings.model_id,
                torch_dtype=self.dtype,
                low_cpu_mem_usage=True,
                use_safetensors=True,
                token=settings.hf_token,
            )
            model.to(self.device)

            self._patch_model_config(model, tokenizer)

            self._pipe = pipeline(
                "automatic-speech-recognition",
                model=model,
                tokenizer=tokenizer,
                feature_extractor=processor.feature_extractor,
                chunk_length_s=settings.chunk_length_s,
                batch_size=settings.batch_size,
                torch_dtype=self.dtype,
                device=self.device,
                token=settings.hf_token,
            )
        except Exception as e:
            print(f"Error: Failed to load pipeline. {e}")
            sys.exit(1)

    def _format_timestamp(self, seconds: float) -> str:
        """Convert seconds to SRT format (HH:MM:SS,mmm)."""
        millis = int((seconds - int(seconds)) * 1000)
        minutes, seconds_int = divmod(int(seconds), 60)
        hours, minutes = divmod(minutes, 60)
        return f"{hours:02}:{minutes:02}:{seconds_int:02},{millis:03}"

    def _write_srt(
        self, chunks: list[dict], output_path: Path, offset: float = 0.0
    ) -> None:
        """Write transcription chunks to SRT file."""
        with open(output_path, "w", encoding="utf-8") as f:
            for idx, chunk in enumerate(chunks, start=1):
                text = chunk.get("text", "").strip()
                timestamp = chunk.get("timestamp")

                if not text or not timestamp:
                    continue

                start, end = timestamp
                if end is None:
                    end = start + 2.0  # Handle EOF

                f.write(f"{idx}\n")
                f.write(
                    f"{self._format_timestamp(start + offset)} --> "
                    f"{self._format_timestamp(end + offset)}\n"
                )
                f.write(f"{text}\n\n")

    def transcribe(
        self,
        audio_path: Path,
        output_path: Path | None = None,
        start_time: float = 0.0,
        end_time: float | None = None,
    ) -> None:
        """Execute the transcription workflow."""
        if not audio_path.exists():
            print(f"Error: File {audio_path} not found.")
            return

        if output_path is None:
            output_path = audio_path.with_suffix(".srt")

        # 1. Subtitle Lookup (Only if processing full file)
        is_full_file = start_time == 0.0 and end_time is None
        if is_full_file:
            if self._extract_existing_subtitles(audio_path, output_path):
                return

        # 2. Prepare Audio
        process_path = audio_path
        if start_time > 0 or end_time is not None:
            process_path = self._slice_audio(audio_path, start_time, end_time)

        # 3. Load Model & Transcribe
        self._load_pipeline()
        if not self._pipe:
            return

        print(f"Info: Transcribing ({settings.language})...")
        try:
            result = self._pipe(
                str(process_path),
                return_timestamps=True,
                generate_kwargs={"language": settings.language, "task": "transcribe"},
            )
        except Exception as e:
            print(f"Error during inference: {e}")
            return
        finally:
            if process_path != audio_path and process_path.exists():
                process_path.unlink()

        # 4. Save Output
        chunks: list[dict] = result.get("chunks", [])  # type: ignore
        self._write_srt(chunks, output_path, offset=start_time)
        print(f"Success: Saved to {output_path}")


def main():
    """CLI Entry point."""
    if len(sys.argv) < 2:
        print("Usage: python transcriber.py <audio_file> [start_time] [end_time]")
        sys.exit(1)

    audio_file = Path(sys.argv[1]).resolve()

    # Simple CLI parsing for time ranges
    start = float(sys.argv[2]) if len(sys.argv) > 2 else 0.0
    end = float(sys.argv[3]) if len(sys.argv) > 3 else None

    app = AudioTranscriber()
    app.transcribe(audio_file, start_time=start, end_time=end)

    # Extract and save plain text (after transcription completes)
    out_path = audio_file.with_suffix(".txt")
    try:
        # Read SRT to extract text lines (skip indices/timestamps)
        with open(audio_file.with_suffix(".srt"), "r", encoding="utf-8") as srt_f:
            lines = [
                line.strip()
                for line in srt_f
                if line.strip()
                and not line.strip().startswith(("-->", "0:", "1:", "2:"))
            ]
        # Filter to text only (odd lines after timestamps)
        text_lines = lines[2::4]  # SRT format: index, time, text, blank

        with open(out_path, "w", encoding="utf-8") as f:
            f.write("\n".join(text_lines) + "\n")

        print(f"Success: Plain text saved to {out_path}")
    except FileNotFoundError:
        print("Warning: No SRT generated, skipping TXT output.")


if __name__ == "__main__":
    main()
