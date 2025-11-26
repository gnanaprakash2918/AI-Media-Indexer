"""Audio transcription utilities using Faster Whisper.

This module provides a small wrapper around Faster Whisper optimized for local
execution with fallback to online download. It adds language-driven model
selection (for Tamil -> `vasista22/whisper-tamil-large-v2`) and optionally
pre-downloads the model using huggingface_hub to show progress and speed up
Faster Whisper initialization.

Usage:
    transcriber = AudioTranscriber()
    result = transcriber.transcribe("path/to/file.mp3", language="ta")

Notes:
    * If you want to use the converted CT2 model for Tamil (recommended to
      improve speed/efficiency), convert and place it under `models/whisper-tamil-ct2`
      and the class will pick it automatically.
    * To force a particular model, pass `model_size` during constructor.
"""

from __future__ import annotations

import os
from pathlib import Path

import torch
from faster_whisper import WhisperModel

from core.schemas import TranscriptionResult

from ...config import settings


class AudioTranscriber:
    """Audio transcriber using Faster Whisper.

    Optimized for local execution with auto-GPU detection and specific
    model caching strategies.
    """Transcribe audio using Faster Whisper with language-driven model choice.

    The transcriber auto-detects a GPU (CUDA) and sets a sensible compute type.
    If the requested language is Tamil ("ta"), it prefers the `vasista22`
    model and will use a local converted model path if present. For English
    or unspecified languages, it defaults to `large-v3`. Additional language
    → model mappings can be added via `Settings.WHISPER_MODEL_MAP`.

    Attributes:
        model_size: The name of the Faster Whisper model or path to a local
            converted model.
        device: Device to load the model on ("cuda" or "cpu").
        compute_type: Numeric precision used by the model ("float16", "int8",
            "float32", etc.).
    """

    def __init__(
        self,
        model_size: str | None = None,
        compute_type: str | None = None,
        device: str | None = None,
    ) -> None:
        """Initialize the Faster Whisper model.

        Args:
            model_size: Model size (e.g., "large-v3" or path to converted model).
            compute_type: Overrides auto-detection ("float16", "int8", "float32").
            device: Overrides auto-detection ("cuda" or "cpu").

        Raises:
            RuntimeError: If the model fails to load.
        """
        self.model_size = model_size or settings.WHISPER_MODEL
        self.device = device or settings.WHISPER_DEVICE
        self.compute_type = compute_type or settings.WHISPER_COMPUTE_TYPE

        self.project_root = self._find_project_root()
        self.model_root_dir = self.project_root / "models"
        self.model_root_dir.mkdir(parents=True, exist_ok=True)

        self._add_torch_libs_to_path(self.project_root)

        if not self.device:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"

        if not self.compute_type:
            self.compute_type = "float16" if self.device == "cuda" else "int8"

        print(
            f"Initializing Whisper: Device={self.device}, "
            f"Compute={self.compute_type}, Model={self.model_size}",
            flush=True,
        )

        try:
            self.model = WhisperModel(
                self.model_size,
                device=self.device,
                compute_type=self.compute_type,
                download_root=str(self.model_root_dir),
            )
        except Exception as exc:
            print(f"critical: Failed to load Whisper model: {exc}", flush=True)
            raise RuntimeError("Could not initialize Whisper") from exc

    def _add_torch_libs_to_path(self, project_root: Path) -> None:
        """Fix specific to Windows + uv/venv to ensure Torch DLLs are found.

        Args:
            project_root: The root directory of the project containing .venv.
        """
        try:
            torch_lib = (
                project_root / ".venv" / "Lib" / "site-packages" / "torch" / "lib"
            )
            if torch_lib.exists():
                os.environ["PATH"] = str(torch_lib) + os.pathsep + os.environ["PATH"]
                print(f"Added Torch DLLs to PATH: {torch_lib}", flush=True)
        except Exception as e:
            print(f" Warning Could not add Torch libs to path: {e}", flush=True)

    def _find_project_root(self) -> Path:
        """Traverse up until we find pyproject.toml, .venv, or .git."""
        current = Path(__file__).resolve()
        for parent in current.parents:
            if any(
                (parent / marker).exists()
                for marker in ["pyproject.toml", ".venv", ".git"]
            ):
                return parent
        return current.parent.parent.parent

    def transcribe(
        self,
        audio_path: str | Path,
        language: str | None = None,
        initial_prompt: str | None = None,
        beam_size: int = 5,
        vad_filter: bool = True,
    ) -> TranscriptionResult:
        """Transcribe audio and return recognized text with timestamps.

        Args:
            audio_path: Path to the input audio file.
            language: Language code (e.g., "ta", "en"). If None, auto-detects.
            initial_prompt: Text to provide context (fixes spellings/context).
            beam_size: Beam search size (higher = better accuracy, slower).
            vad_filter: Whether to filter out silence.

        Returns:
            TranscriptionResult: A Pydantic model containing text and metadata.

        Raises:
            FileNotFoundError: If audio file does not exist.
            RuntimeError: If transcription fails.
        """
        path_obj = Path(audio_path)
        if not path_obj.exists():
            raise FileNotFoundError(f"Audio file not found: {path_obj}")

        print(
            f"info: Transcribing: {path_obj.name} | Lang: {language or 'Auto'} | "
            f"Prompt: {bool(initial_prompt)}",
            flush=True,
        )

        try:
            segments_generator, info = self.model.transcribe(
                str(path_obj),
                beam_size=beam_size,
                vad_filter=vad_filter,
                vad_parameters={"min_silence_duration_ms": 500} if vad_filter else None,
                task="transcribe",
                language=language,
                initial_prompt=initial_prompt,
            )

            segments = list(segments_generator)
            full_text = "".join([s.text for s in segments]).strip()

            result = TranscriptionResult(
                text=full_text,
                segments=[
                    {"start": s.start, "end": s.end, "text": s.text.strip()}
                    for s in segments
                ],
                language=info.language,
                language_probability=info.language_probability,
                duration=info.duration,
            )

            print(
                f"Complete. Lang: {info.language} ({info.language_probability:.2f}), "
                f"Duration: {info.duration:.2f}s",
                flush=True,
            )
            return result

        except Exception as e:
            print(f"error: Transcription failed for {path_obj.name}: {e}", flush=True)
            raise RuntimeError(f"Transcription failed: {e}") from e


def main() -> None:
    """Manual test harness."""
    print("Script Started", flush=True)

    test_path = Path(r"C:\\Users\\Gnana Prakash M\\Downloads\\Music\\clip.mp3")

    if not test_path.exists():
        print(f"warning: Test file not found at {test_path}", flush=True)
        return

    try:
        transcriber = AudioTranscriber()

        res = transcriber.transcribe(
            test_path,
            language="ta",  # Specify "ta" for Tamil, "en" for English
            initial_prompt="இது ஒரு தமிழ் ஆடியோ பதிவு.",  # Helps with context
            beam_size=5,  # Higher beam_size = better accuracy
        )

        print("\nFinal Output (Pydantic Model)", flush=True)
        print(f"Text Snippet: {res.text[:1000]}...", flush=True)
        print(f"Detected Language: {res.language}", flush=True)
    except Exception as e:
        print(f"error: Test run failed: {e}", flush=True)


if __name__ == "__main__":
    main()
