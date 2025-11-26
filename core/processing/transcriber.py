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
from huggingface_hub import snapshot_download

from config import settings
from core.schemas import TranscriptionResult


class AudioTranscriber:
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
        predownload: bool = True,
    ) -> None:
        """Initialize the transcriber.

        Args:
            model_size: Explicit model ID or local path. If None, model is chosen
                based on settings and language mappings.
            compute_type: Precision override (e.g., "float16", "int8",
                "float32"). If None it is inferred from device.
            device: Device override ("cuda" or "cpu"). If None it is autodetected.
            predownload: If True, attempt to pre-download the HF model with
                progress to `models/` to speed up Faster Whisper load.

        Raises:
            RuntimeError: If the Whisper model fails to load.
        """
        self.model_size = model_size or settings.WHISPER_MODEL
        self.device = device or settings.WHISPER_DEVICE
        self.compute_type = compute_type or settings.WHISPER_COMPUTE_TYPE

        # Placeholder for loaded model and its id (populated on first load).
        self.model: WhisperModel | None = None
        self.model_id: str | None = None

        self.project_root = self._find_project_root()
        self.model_root_dir = self.project_root / "models"
        self.model_root_dir.mkdir(parents=True, exist_ok=True)

        # Windows-specific torch DLL fix (keeps structure intact).
        self._add_torch_libs_to_path(self.project_root)

        if not self.device:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"

        if not self.compute_type:
            # Use float16 on CUDA, int8 on CPU as a reasonable default.
            self.compute_type = "float16" if self.device == "cuda" else "int8"

        print(
            f"Initializing Whisper: Device={self.device}, "
            f"Compute={self.compute_type}, Model={self.model_size}",
            flush=True,
        )

        # If the configured model_size is a mapping placeholder, leave as-is.
        # The actual model will be loaded later when a language is known.
        self.model = None

        # Optionally predownload the model (no model loaded yet)
        self._predownload_enabled = bool(predownload)

    def _add_torch_libs_to_path(self, project_root: Path) -> None:
        """Add Torch DLL directory to PATH on Windows virtualenvs.

        Args:
            project_root: Project root path that may contain `.venv`.
        """
        try:
            torch_lib = (
                project_root / ".venv" / "Lib" / "site-packages" / "torch" / "lib"
            )
            if torch_lib.exists():
                os.environ["PATH"] = (
                    str(torch_lib) + os.pathsep + os.environ.get("PATH", "")
                )

                print(f"Added Torch DLLs to PATH: {torch_lib}", flush=True)
        except Exception as exc:
            print(f"Warning: Could not add Torch libs to path: {exc}", flush=True)

    def _find_project_root(self) -> Path:
        """Find the project root by walking parents for pyproject.toml, .venv, or .git.

        Returns:
            Path pointing at the project root. Falls back to ancestor if none found.
        """
        current = Path(__file__).resolve()
        for parent in current.parents:
            if any(
                (parent / marker).exists()
                for marker in ["pyproject.toml", ".venv", ".git"]
            ):
                return parent
        return current.parent.parent.parent

    def _model_for_language(self, language: str | None) -> str:
        """Choose model ID or path based on language and settings.

        Args:
            language: ISO language code, e.g., "ta", "en".

        Returns:
            Model id or local path to model directory.
        """
        # If explicit model_size passed in constructor, it always wins.
        if self.model_size and self.model_size != settings.WHISPER_MODEL:
            return self.model_size

        lang = (language or "").lower()
        mapping = settings.WHISPER_MODEL_MAP or {}
        model_choice = mapping.get(lang) or settings.WHISPER_MODEL or "large-v3"

        # Hardcoded fallback for Tamil to use local converted model if present.
        if (
            model_choice.startswith("vasista22")
            and (self.model_root_dir / "whisper-tamil-ct2").exists()
        ):
            local_path = str(self.model_root_dir / "whisper-tamil-ct2")
            print(f"Using local converted Tamil model at: {local_path}", flush=True)
            return local_path

        return model_choice

    def _maybe_predownload(self, model_id: str) -> None:
        """Attempt to pre-download the model files to the `models/` directory.

        This uses huggingface_hub.snapshot_download when available to show progress
        and improve downstream load time for Faster Whisper.

        Args:
            model_id: HF model id, for example:
                "vasista22/whisper-tamil-large-v2" or
                "openai/whisper-large-v3".
        """
        if not snapshot_download:
            # huggingface_hub not installed
            print("huggingface_hub not available; skipping pre-download.", flush=True)
            return

        # Only attempt if model_id looks like a HF repo (contains a slash).
        if "/" not in model_id:
            return

        try:
            target_cache = str(self.model_root_dir / model_id.replace("/", "_"))
            print(
                f"Attempting to pre-download '{model_id}' into {target_cache}",
                flush=True,
            )

            snapshot_download(repo_id=model_id, cache_dir=str(self.model_root_dir))
            print(f"Pre-download complete for {model_id}", flush=True)
        except Exception as exc:
            print(f"Warning: pre-download failed for {model_id}: {exc}", flush=True)

    def _load_whisper_model(self, model_id: str) -> WhisperModel:
        """Instantiate and return a WhisperModel from faster_whisper.

        Args:
            model_id: Model id or local path.

        Returns:
            WhisperModel instance.

        Raises:
            RuntimeError: If WhisperModel initialization fails.
        """
        # If predownload enabled, attempt to predownload (best-effort).
        if self._predownload_enabled:
            self._maybe_predownload(model_id)

        try:
            device = self.device or ("cuda" if torch.cuda.is_available() else "cpu")
            compute = (
                self.compute_type
                if self.compute_type
                else ("float16" if device == "cuda" else "int8")
            )

            model = WhisperModel(
                model_id,
                device=device,
                compute_type=compute,
                download_root=str(self.model_root_dir),
            )
            return model
        except Exception as exc:
            print(
                f"critical: Failed to load Whisper model '{model_id}': {exc}",
                flush=True,
            )
            raise RuntimeError(
                f"Could not initialize Whisper model '{model_id}'"
            ) from exc

    def transcribe(
        self,
        audio_path: str | Path,
        language: str | None = None,
        initial_prompt: str | None = None,
        beam_size: int = 5,
        vad_filter: bool = True,
    ) -> TranscriptionResult:
        """Transcribe an audio file and return recognized text with timestamps.

        This method will select an appropriate model based on the provided
        `language`. If the model for the language is not yet loaded, it will
        be loaded now (with a pre-download step if enabled).

        Args:
            audio_path: Path to the input audio file.
            language: Language code (e.g., "ta", "en"). If None, autodetect.
            initial_prompt: A text prompt used by the model to bias recognition.
            beam_size: Beam search size (higher may improve accuracy at cost of speed).
            vad_filter: Whether to filter out silence using VAD.

        Returns:
            TranscriptionResult: Pydantic model containing text, segments, language
                and metadata.

        Raises:
            FileNotFoundError: If the audio file does not exist.
            RuntimeError: If transcription fails.
        """
        path_obj = Path(audio_path)
        if not path_obj.exists():
            raise FileNotFoundError(f"Audio file not found: {path_obj}")

        # Determine model based on language preference
        chosen_model = self._model_for_language(language)
        print(
            (
                f"info: Transcribing '{path_obj.name}' | Lang: {language or 'Auto'}"
                f" | Model: {chosen_model}"
            ),
            flush=True,
        )

        # Load model now (if not loaded or different)
        if (
            getattr(self, "model", None) is None
            or getattr(self, "model_id", None) != chosen_model
        ):
            self.model = self._load_whisper_model(chosen_model)
            self.model_id = chosen_model

        try:
            if self.model is None:
                raise RuntimeError(
                    "Whisper model not loaded. Call transcribe with a valid "
                    "model available."
                )

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
                language=getattr(info, "language", language or ""),
                language_probability=getattr(info, "language_probability", 0.0),
                duration=getattr(info, "duration", 0.0),
            )

            print(
                (
                    f"Complete. Lang: {result.language} "
                    f"({result.language_probability:.2f}), "
                    f"Duration: {result.duration:.2f}s"
                ),
                flush=True,
            )

            return result

        except Exception as exc:
            print(f"error: Transcription failed for {path_obj.name}: {exc}", flush=True)
            raise RuntimeError(f"Transcription failed: {exc}") from exc


def main() -> None:
    """Small manual test harness.

    Edit `test_path` to point to a real audio file on your machine to test.
    """
    print("Script Started", flush=True)

    test_path = Path(r"C:\Users\Gnana Prakash M\Downloads\Music\clip.mp3")

    if not test_path.exists():
        print(f"warning: Test file not found at {test_path}", flush=True)
        return

    try:
        # The transcriber will pick Tamil model when language="ta"
        transcriber = AudioTranscriber(predownload=True)

        res = transcriber.transcribe(
            test_path,
            language="ta",  # "ta" will pick the Tamil model mapping
            initial_prompt="இது ஒரு தமிழ் ஆடியோ பதிவு.",  # Helps with Tamil context
            beam_size=5,
        )

        print("\nFinal Output (Pydantic Model)", flush=True)
        print(f"Text Snippet: {res.text[:1000]}...", flush=True)
        print(f"Detected Language: {res.language}", flush=True)
    except Exception as exc:
        print(f"error: Test run failed: {exc}", flush=True)


if __name__ == "__main__":
    main()
