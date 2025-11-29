"""Robust Audio Transcriber with Subtitle Fallback and HF Patching.

Features:
- Smart Caching: Project Local > Global Cache > Download.
- Prioritizes existing subtitles (User > Sidecar > Embedded).
- CRITICAL FIX: Replaces outdated generation configs for fine-tuned models.
- Dynamic Model Selection based on Language.
"""

import gc
import math
import shutil
import subprocess
import sys
import warnings
from pathlib import Path
from tempfile import NamedTemporaryFile
from typing import Any, List, Optional

import torch
from huggingface_hub import snapshot_download
from transformers import (
    AutoModelForSpeechSeq2Seq,
    AutoProcessor,
    AutoTokenizer,
    GenerationConfig,  # type: ignore
)
from transformers.pipelines import pipeline
from transformers.utils import logging as hf_logging

from config import settings

# Clean up console output
warnings.filterwarnings("ignore")
hf_logging.set_verbosity_error()


class AudioTranscriber:
    """Main transcription class handling ASR lifecycle."""

    def __init__(self) -> None:
        """Initialize transcriber."""
        self._pipe: Any = None
        self._current_model_id: Optional[str] = None
        print(
            "[INFO] Initialized Transcriber "
            f"(Device: {settings.device}, Index: {settings.device_index})"
        )

    def _get_ffmpeg_cmd(self) -> str:
        """Locate FFmpeg executable."""
        cmd = shutil.which("ffmpeg")
        if not cmd:
            raise RuntimeError("FFmpeg not found. Please install it to system PATH.")
        return cmd

    def _extract_existing_subtitles(
        self,
        input_path: Path,
        output_path: Path,
        user_sub_path: Optional[Path],
        language: str,
    ) -> bool:
        """Try to find/extract subtitles from various sources."""
        print("[INFO] Probing for existing subtitles...")

        # 1. User Provided
        if user_sub_path and user_sub_path.exists():
            print(f"[SUCCESS] Using provided subtitle: {user_sub_path}")
            shutil.copy(user_sub_path, output_path)
            return True

        # 2. Sidecar Files
        candidates = [
            input_path.with_suffix(f".{language}.srt"),
            input_path.with_suffix(".srt"),
        ]
        for sidecar in candidates:
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

    def _slice_audio(
        self, input_path: Path, start: float, end: Optional[float]
    ) -> Path:
        """Slice audio safely using a named temporary file."""
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
        subprocess.run(cmd, check=True)
        return output_slice

    def _download_model(self, model_id: str) -> Path:
        """Smart Download: Local Project -> Global Cache -> Download to Local."""
        # Check Local
        try:
            return Path(
                snapshot_download(
                    repo_id=model_id,
                    cache_dir=settings.model_cache_dir,
                    token=settings.hf_token,
                    local_files_only=True,
                )
            )
        except Exception:
            pass

        # Check Global
        try:
            print(f"[INFO] Checking global cache for '{model_id}'...")
            return Path(
                snapshot_download(
                    repo_id=model_id,
                    cache_dir=None,
                    token=settings.hf_token,
                    local_files_only=True,
                )
            )
        except Exception:
            pass

        # Download
        print(f"[INFO] Downloading '{model_id}' to '{settings.model_cache_dir}'...")
        try:
            return Path(
                snapshot_download(
                    repo_id=model_id,
                    cache_dir=settings.model_cache_dir,
                    token=settings.hf_token,
                    resume_download=True,
                )
            )
        except Exception as e:
            print(f"[ERROR] Download failed: {e}")
            sys.exit(1)

    def _patch_model_config(self, model: Any, tokenizer: Any, language: str) -> None:
        """Replace broken generation configs with clean standard ones."""
        # We MUST load a fresh config from the base architecture (OpenAI).
        try:
            # Determine base model size (Vasista is Large-v2)
            base_config_id = "openai/whisper-large-v2"
            if "large-v3" in getattr(model.config, "_name_or_path", "").lower():
                base_config_id = "openai/whisper-large-v3"

            print(f"[DEBUG] Forcing clean GenerationConfig from '{base_config_id}'...")

            # Load clean config (Transformers handles caching this automatically)
            clean_config = GenerationConfig.from_pretrained(base_config_id)
            model.generation_config = clean_config

        except Exception as e:
            print(
                f"[WARN] Failed to load base config: {e}. Using existing (may crash)."
            )
        # --- CRITICAL FIX END ---

        gen_config = model.generation_config

        # 1. Fix 'no_timestamps_token_id'
        if not getattr(gen_config, "no_timestamps_token_id", None):
            notimestamp_token = "<|notimestamps|>"
            token_id = tokenizer.convert_tokens_to_ids(notimestamp_token)
            unk_id = getattr(tokenizer, "unk_token_id", None)
            if token_id is not None and token_id != unk_id:
                gen_config.no_timestamps_token_id = token_id  # type: ignore

        # 2. Clear forced_decoder_ids (Transformers will regenerate it for target lang)
        if hasattr(gen_config, "forced_decoder_ids"):
            gen_config.forced_decoder_ids = None  # type: ignore

        # 3. Explicitly set Language & Task
        gen_config.language = language  # type: ignore
        gen_config.task = "transcribe"  # type: ignore
        gen_config.is_multilingual = True  # type: ignore

    def _load_pipeline(self, language: str) -> None:
        """Load ASR pipeline dynamically."""
        candidates = settings.whisper_model_map.get(language, [])
        model_id = candidates[0] if candidates else settings.fallback_model_id

        if self._pipe and self._current_model_id == model_id:
            return

        if self._pipe:
            print("[INFO] Switching models... clearing VRAM.")
            del self._pipe
            gc.collect()
            torch.cuda.empty_cache()
            self._pipe = None

        snapshot_dir = self._download_model(model_id)
        print(f"[INFO] Loading '{model_id}' from {snapshot_dir}...")

        try:
            tokenizer = AutoTokenizer.from_pretrained(snapshot_dir)
            processor = AutoProcessor.from_pretrained(snapshot_dir)

            # Using standard loading (removed use_safetensors to fix previous error)
            model = AutoModelForSpeechSeq2Seq.from_pretrained(
                snapshot_dir,
                torch_dtype=settings.torch_dtype,
                low_cpu_mem_usage=True,
            )
            model.to(settings.device)

            # Apply the Critical Fix
            self._patch_model_config(model, tokenizer, language)

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
        except Exception as e:
            print(f"[ERROR] Failed to load pipeline: {e}")
            sys.exit(1)

    def _format_timestamp(self, seconds: float) -> str:
        """SRT Timestamp format."""
        hours = int(seconds // 3600)
        minutes = int((seconds % 3600) // 60)
        secs = seconds % 60
        millis = int(round((secs - math.floor(secs)) * 1000))
        if millis == 1000:
            secs += 1
            millis = 0
        return f"{hours:02}:{minutes:02}:{int(secs):02},{millis:03}"

    def _write_srt(self, chunks: List[dict], output_path: Path, offset: float) -> None:
        """Write SRT file."""
        with open(output_path, "w", encoding="utf-8") as f:
            for idx, chunk in enumerate(chunks, start=1):
                text = chunk.get("text", "").strip()
                timestamp = chunk.get("timestamp")
                if not text or not timestamp:
                    continue

                start, end = timestamp
                if end is None:
                    end = start + 2.0

                f.write(
                    f"{idx}\n{self._format_timestamp(start + offset)} --> "
                    f"{self._format_timestamp(end + offset)}\n{text}\n\n"
                )

    def transcribe(
        self,
        audio_path: Path,
        language: Optional[str] = None,
        subtitle_path: Optional[Path] = None,
        output_path: Optional[Path] = None,
        start_time: float = 0.0,
        end_time: Optional[float] = None,
    ) -> None:
        """Execute transcription."""
        if not audio_path.exists():
            print(f"[ERROR] File {audio_path} not found.")
            return

        lang = language if language else settings.language
        out = output_path if output_path else audio_path.with_suffix(".srt")

        # 1. Check Subtitles
        if start_time == 0.0 and end_time is None:
            if self._extract_existing_subtitles(audio_path, out, subtitle_path, lang):
                return

        # 2. Slice Audio
        proc_path = audio_path
        is_sliced = False
        if start_time > 0 or end_time is not None:
            proc_path = self._slice_audio(audio_path, start_time, end_time)
            is_sliced = True

        # 3. Transcribe
        self._load_pipeline(lang)
        if not self._pipe:
            return

        print(f"[INFO] Transcribing (Lang: {lang})...")
        try:
            res = self._pipe(
                str(proc_path),
                return_timestamps=True,
                generate_kwargs={"language": lang, "task": "transcribe"},
            )
            self._write_srt(res.get("chunks", []), out, offset=start_time)  # type: ignore
            print(f"[SUCCESS] Saved to {out}")
        except Exception as e:
            print(f"[ERROR] Inference failed: {e}")
        finally:
            if is_sliced and proc_path.exists():
                try:
                    proc_path.unlink()
                except:
                    pass


def main() -> None:
    """Command-line interface for AudioTranscriber."""
    if len(sys.argv) < 2:
        print(
            "Usage: "
            "python -m core.processing.transcriber "
            "<audio_file> [start] [end] [subtitle_file] [lang]"
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
    app.transcribe(
        audio, language=lang, subtitle_path=sub, start_time=start, end_time=end
    )


if __name__ == "__main__":
    main()
