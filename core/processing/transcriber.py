"""Audio transcription utilities using Faster Whisper.

This module provides a small wrapper around Faster Whisper optimized for
local execution with fallback to online download.
"""

from __future__ import annotations

import os
from pathlib import Path

import torch
from faster_whisper import WhisperModel

from core.schemas import TranscriptionResult


class AudioTranscriber:
    """Audio transcriber using Faster Whisper.

    Optimized for local execution with auto-GPU detection and specific
    model caching strategies.
    """

    def __init__(
        self,
        model_size: str = "large-v2",
        compute_type: str | None = None,
        device: str | None = None,
    ) -> None:
        """Initialize the Faster Whisper model.

        Args:
            model_size: Model size (e.g., "base", "small", "large-v2").
            compute_type: Overrides auto-detection ("float16", "int8", "float32").
            device: Overrides auto-detection ("cuda" or "cpu").

        Raises:
            RuntimeError: If the model fails to load.
        """
        self.model_size = model_size
        self.project_root = self._find_project_root()

        self.model_root_dir = self.project_root / "models"
        self.model_root_dir.mkdir(parents=True, exist_ok=True)

        self._add_torch_libs_to_path(self.project_root)

        if device:
            self.device = device
        else:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"

        if compute_type:
            self.compute_type = compute_type
        else:
            # Fallback logic: int8 is faster on CPU, float16 best for CUDA
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

    def transcribe(self, audio_path: str | Path) -> TranscriptionResult:
        """Transcribe audio and return recognized text with timestamps.

        Args:
            audio_path: Path to the input audio file.

        Returns:
            TranscriptionResult: A Pydantic model containing text and metadata.

        Raises:
            FileNotFoundError: If audio file does not exist.
            RuntimeError: If transcription fails.
        """
        path_obj = Path(audio_path)
        if not path_obj.exists():
            raise FileNotFoundError(f"Audio file not found: {path_obj}")

        print(f"info: Transcribing: {path_obj.name}...", flush=True)

        try:
            # beam_size=5 provides better accuracy
            segments_generator, info = self.model.transcribe(
                str(path_obj),
                beam_size=5,
                vad_filter=True,  # Remove silence
                vad_parameters={"min_silence_duration_ms": 500},
                task="transcribe",
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
        # Use 'base' or 'tiny' for quick testing. Use 'large-v2' for prod.
        transcriber = AudioTranscriber(model_size="large-v3")
        res = transcriber.transcribe(test_path)

        print("\nFinal Output (Pydantic Model)", flush=True)
        print(f"Text Snippet: {res.text[:1000]}...", flush=True)
        print(f"Detected Language: {res.language}", flush=True)
    except Exception as e:
        print(f"error: Test run failed: {e}", flush=True)


if __name__ == "__main__":
    main()
