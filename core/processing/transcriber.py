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
import shutil
import sys
from pathlib import Path

os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "1"

import torch
from ctranslate2.converters import TransformersConverter
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

        self.model: WhisperModel | None = None
        self.model_id: str | None = None

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
        )

        self.model = None
        self._predownload_enabled = bool(predownload)

    def _ensure_ct2_model(
        self, hf_model_id: str, ct2_dir: Path, quantization: str = "float16"
    ) -> bool:
        """Ensure CT2 model exists, convert if missing using Direct Python API."""
        if ct2_dir.exists() and (ct2_dir / "model.bin").exists():
            return True

        if TransformersConverter is None:
            print("Error: ctranslate2 not installed or import failed.")
            return False

        print(
            f"CT2 model not found at {ct2_dir}. Starting conversion pipeline...",
        )

        # 1. Download RAW model first (High Speed)
        sanitized_name = hf_model_id.replace("/", "_")
        raw_model_dir = self.model_root_dir / f"raw_{sanitized_name}"

        try:
            print(f"Phase 1: High-speed download of {hf_model_id}...")
            snapshot_download(
                repo_id=hf_model_id,
                local_dir=raw_model_dir,
                ignore_patterns=["*.msgpack", "*.h5", "*.tflite", "*.ot"],
            )
        except Exception as e:
            print(f"Download failed: {e}")
            return False

        # 2. Convert LOCAL raw model to CT2 (Direct Call)
        print("Phase 2: Converting local raw model to CT2 format...")
        try:
            # Initialize the converter with the path we just downloaded
            converter = TransformersConverter(
                str(raw_model_dir), low_cpu_mem_usage=True
            )

            # Run the conversion
            converter.convert(str(ct2_dir), quantization=quantization, force=True)

            print("CT2 conversion complete.")

            # Optional: Remove raw model to save space
            shutil.rmtree(raw_model_dir)

            return ct2_dir.exists()

        except Exception as exc:
            print(f"CT2 conversion failed with error: {exc}")
            # Cleanup failed conversion dir
            if ct2_dir.exists():
                shutil.rmtree(ct2_dir)
            return False

    def _add_torch_libs_to_path(self, project_root: Path) -> None:
        try:
            torch_lib = (
                project_root / ".venv" / "Lib" / "site-packages" / "torch" / "lib"
            )
            if torch_lib.exists():
                os.environ["PATH"] = (
                    str(torch_lib) + os.pathsep + os.environ.get("PATH", "")
                )

                if str(torch_lib) not in sys.path:
                    sys.path.append(str(torch_lib))
        except Exception:
            print("warning: Could not add torch libs to PATH.")
            pass

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
        mapping = getattr(settings, "WHISPER_MODEL_MAP", None) or {}
        model_choice = mapping.get(lang) or settings.WHISPER_MODEL or "large-v3"

        # Fallback for Tamil
        if model_choice.startswith("vasista22"):
            ct2_dir = self.model_root_dir / "whisper-tamil-ct2"

            if not ct2_dir.exists() or not (ct2_dir / "model.bin").exists():
                self._ensure_ct2_model(
                    model_choice,
                    ct2_dir,
                    quantization="float16" if self.device == "cuda" else "int8",
                )

        """Attempt to pre-download the model files to the `models/` directory."""
        if ct2_dir.exists():
            local_path = str(ct2_dir)
            print(f"Using local converted Tamil model at: {local_path}")
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
            return
        if "/" not in model_id:
            return

        token = os.getenv("HF_TOKEN") or os.getenv("HUGGINGFACE_HUB_TOKEN") or None

        allow_patterns = [
            "config.json",
            "generation_config.json",
            "*.bin",
            "*.pt",
            "*.onnx",
            "*.safetensors",
            "pytorch_model*.bin",
            "model.safetensors",
            "tokenizer.json",
            "vocab.json",
            "tokenizer_config.json",
            "preprocessor_config.json",
        ]

        try:
            snapshot_download(
                repo_id=model_id,
                allow_patterns=allow_patterns,
                token=token,
            )
        except Exception:
            pass

    def _load_whisper_model(self, model_id: str) -> WhisperModel:
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

        chosen_model = self._model_for_language(language)
        print(
            (
                f"info: Transcribing '{path_obj.name}' | Lang: {language or 'Auto'}"
                f" | Model: {chosen_model}"
            ),
        )

        if (
            getattr(self, "model", None) is None
            or getattr(self, "model_id", None) != chosen_model
        ):
            self.model = self._load_whisper_model(chosen_model)
            self.model_id = chosen_model

        try:
            if self.model is None:
                raise RuntimeError("Whisper model not loaded.")

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
            )

            return result

        except Exception as exc:
            print(f"error: Transcription failed for {path_obj.name}: {exc}")
            raise RuntimeError(f"Transcription failed: {exc}") from exc


def main() -> None:
    """Small manual test harness.

    Edit `test_path` to point to a real audio file on your machine to test.
    """
    print("Script Started")
    test_path = Path(r"C:\Users\Gnana Prakash M\Downloads\Music\clip.mp3")
    if not test_path.exists():
        print(f"warning: Test file not found at {test_path}")
        return

    try:
        transcriber = AudioTranscriber(predownload=True)
        res = transcriber.transcribe(
            test_path,
            language="ta",
            initial_prompt="இது ஒரு தமிழ் ஆடியோ பதிவு.",
            beam_size=5,
        )
        print("\nFinal Output (Pydantic Model)")
        print(f"Text Snippet: {res.text[:1000]}...")
        print(f"Detected Language: {res.language}")
    except Exception as exc:
        print(f"error: Test run failed: {exc}")


if __name__ == "__main__":
    main()
