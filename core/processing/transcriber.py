"""Audio transcription module using Faster-Whisper with multi-backend support.

This module implements a robust transcription service that defaults to
Faster-Whisper for speed and local execution but supports Transformers
and NVIDIA NeMo for specific SOTA Tamil models when required.
"""

from __future__ import annotations

import os
import warnings
from pathlib import Path
from typing import Any, Literal

import torch
from faster_whisper import WhisperModel
from huggingface_hub import hf_hub_download, list_repo_files, snapshot_download

from config import settings

warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "1"


class AudioTranscriber:
    """Handles audio transcription with automatic backend selection.

    This class manages the loading of different ASR backends (Faster-Whisper,
    Transformers, NeMo) based on the requested model type and handles the
    transcription process, including language detection and segmentation.

    Attributes:
        device (str): The computing device ('cuda' or 'cpu').
        compute_type (str): The quantization type (e.g., 'float16', 'int8').
        model_root (Path): Directory where models are stored.
        model_id (str): The ID or path of the loaded model.
        model (Any): The loaded model instance (type varies by backend).
        engine (Literal): The active backend engine.
        generation_config (Any): Patched config for Transformers.
    """

    def __init__(
        self,
        model_size: str = "base",
        device: str | None = None,
        compute_type: str | None = None,
        download_root: str | Path | None = None,
    ) -> None:
        """Initialize the transcriber and load the default model.

        Args:
            model_size: Name of the model (e.g., 'base', 'small', 'large-v3')
                or a HuggingFace path. Defaults to "base".
            device: 'cuda' or 'cpu'. Auto-detected if None.
            compute_type: 'float16', 'int8_float16', or 'int8'. Auto-selected
                based on device if None.
            download_root: Custom path to store models. If None, uses
                project_root/models.
        """
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.compute_type = compute_type or (
            "float16" if self.device == "cuda" else "int8"
        )

        if download_root:
            self.model_root = Path(download_root)
        else:
            self.model_root = self._find_project_root() / "models"
        self.model_root.mkdir(parents=True, exist_ok=True)

        self.model_id = model_size
        self.model: Any = None
        self.generation_config: Any = None
        self.engine: Literal["faster_whisper", "transformers", "nemo"] = (
            "faster_whisper"
        )

        # Determine engine and load
        self._load_model(self.model_id)

    def transcribe(
        self,
        audio_path: str | Path,
        language: str | None = None,
        beam_size: int = 5,
        vad_filter: bool = True,
    ) -> dict[str, Any]:
        """Transcribe audio file and return text with segments.

        Args:
            audio_path: Path to the audio file.
            language: Language code (e.g., 'en', 'ta'). Auto-detects if None.
            beam_size: Beam size for decoding. Defaults to 5.
            vad_filter: Whether to filter silence (VAD). Defaults to True.

        Returns:
            dict[str, Any]: A dictionary containing:
                - text (str): The full transcribed text.
                - segments (list): List of segment dicts with start, end, and text.
                - language (str): The detected or specified language code.
                - duration (float): The total duration of the audio.

        Raises:
            FileNotFoundError: If the audio file does not exist.
        """
        path_obj = Path(audio_path)
        if not path_obj.exists():
            raise FileNotFoundError(f"Audio file not found: {path_obj}")

        effective_lang = language
        print(f"info: Transcribing '{path_obj.name}' using {self.engine.upper()}...")

        if self.engine == "nemo":
            return self._transcribe_nemo(path_obj)
        if self.engine == "transformers":
            return self._transcribe_transformers(path_obj, effective_lang)

        return self._transcribe_faster_whisper(
            path_obj, effective_lang, beam_size, vad_filter
        )

    def _transcribe_faster_whisper(
        self, path: Path, language: str | None, beam_size: int, vad_filter: bool
    ) -> dict[str, Any]:
        """Execute transcription using the Faster-Whisper backend.

        Args:
            path: Path to the audio file.
            language: Language code.
            beam_size: Beam size for decoding.
            vad_filter: Whether to apply VAD filtering.

        Returns:
            dict[str, Any]: Transcription result with text, segments, and metadata.
        """
        segments_generator, info = self.model.transcribe(
            str(path),
            beam_size=beam_size,
            language=language,
            vad_filter=vad_filter,
            vad_parameters={"min_silence_duration_ms": 500} if vad_filter else None,
        )

        detected_lang = info.language
        print(
            f"info: Language detected: '{detected_lang}' "
            f"(Prob: {info.language_probability:.2f})"
        )

        segments_list = []
        full_text_parts = []

        for segment in segments_generator:
            start, end, text = segment.start, segment.end, segment.text.strip()
            print(f"  [{start:.2f}s -> {end:.2f}s] {text}")
            segments_list.append({"start": start, "end": end, "text": text})
            full_text_parts.append(text)

        return {
            "text": " ".join(full_text_parts),
            "segments": segments_list,
            "language": detected_lang,
            "duration": info.duration,
        }

    def _load_model(self, model_id: str) -> None:
        """Determines the correct engine and loads the model.

        Args:
            model_id: The model identifier string.
        """
        print(f"info: Loading model '{model_id}' on {self.device.upper()}...")
        self.engine = self._determine_engine(model_id)

        if self.engine == "nemo":
            self._load_nemo(model_id)
        elif self.engine == "transformers":
            self._load_transformers(model_id)
        else:
            self._load_faster_whisper(model_id)

    def _determine_engine(
        self, model_id: str
    ) -> Literal["faster_whisper", "transformers", "nemo"]:
        """Determine which backend engine to use based on the model ID.

        Args:
            model_id: The model identifier string.

        Returns:
            Literal: The engine name ('faster_whisper', 'transformers', or 'nemo').
        """
        lower = model_id.lower()
        if "indicconformer" in lower or "ai4bharat" in lower:
            if "rnnt" in lower or "nemo" in lower:
                return "nemo"
            return "transformers"
        if "jiviai" in lower or "vasista" in lower or "audiox" in lower:
            return "transformers"
        return "faster_whisper"

    def _load_faster_whisper(self, model_id: str) -> None:
        """Load a Faster-Whisper model.

        Args:
            model_id: The model identifier.

        Raises:
            RuntimeError: If model loading fails.
        """
        try:
            self.model = WhisperModel(
                model_id,
                device=self.device,
                compute_type=self.compute_type,
                download_root=str(self.model_root),
            )
        except Exception as e:
            raise RuntimeError(
                f"Failed to load Faster-Whisper model '{model_id}': {e}"
            ) from e

    def _load_transformers(self, model_id: str) -> None:
        """Load a Transformers pipeline model.

        Args:
            model_id: The model identifier.

        Raises:
            ImportError: If dependencies are missing.
        """
        try:
            from transformers import AutoConfig, GenerationConfig
            from trasnformers.pipeline import pipeline
        except ImportError:
            raise ImportError(
                "Please install 'transformers', 'accelerate' to use this model."
            )

        print(f"info: Ensuring '{model_id}' is available in {self.model_root}...")
        try:
            local_model_path = snapshot_download(
                repo_id=model_id,
                token=getattr(settings, "HF_TOKEN", None),
                cache_dir=str(self.model_root),
            )
        except Exception as e:
            print(f"warning: Snapshot download failed: {e}. Falling back to hub ID.")
            local_model_path = model_id

        print(f"info: Initializing Transformers pipeline from {local_model_path}...")
        device_arg = 0 if self.device == "cuda" else -1
        torch_dtype = torch.float16 if self.device == "cuda" else torch.float32

        try:
            self.model = pipeline(
                "automatic-speech-recognition",
                model=local_model_path,
                device=device_arg,
                torch_dtype=torch_dtype,
                token=getattr(settings, "HF_TOKEN", None),
            )
        except Exception as e:
            raise RuntimeError(f"Pipeline init failed: {e}") from e

        # Many fine-tuned models (like vasista22) have broken config files
        # that prevent timestamps from working in newer Transformers.
        try:
            print("info: Applying config patch for timestamps...")

            # 1. Load the "Healthy" Base Configs from OpenAI
            base_gen_config = GenerationConfig.from_pretrained(
                "openai/whisper-large-v2", cache_dir=str(self.model_root)
            )
            base_config = AutoConfig.from_pretrained(
                "openai/whisper-large-v2", cache_dir=str(self.model_root)
            )

            model_instance = self.model.model
            model_instance.generation_config = base_gen_config

            if hasattr(base_config, "alignment_heads"):
                model_instance.config.alignment_heads = base_config.alignment_heads

            model_instance.config.return_timestamps = True

            self.generation_config = base_gen_config

            print(
                "info: Config patched successfully using openai/whisper-large-v2 defaults."
            )

        except Exception as e:
            print(f"warning: Config patch failed: {e}")
            print("warning: Timestamps might fail if config is broken.")

    def _load_nemo(self, model_id: str) -> None:
        """Load an NVIDIA NeMo model."""
        try:
            import nemo.collections.asr as nemo_asr
        except ImportError as e:
            raise ImportError(
                "NVIDIA NeMo is not installed. Please install it for AI4Bharat models."
            ) from e

        print("info: Fetching NeMo model file...")
        try:
            files = list_repo_files(model_id, token=settings.HF_TOKEN)
            nemo_file = next((f for f in files if f.endswith(".nemo")), None)

            if not nemo_file:
                self.model = nemo_asr.models.ASRModel.from_pretrained(
                    model_name=model_id
                )
            else:
                ckpt = hf_hub_download(
                    model_id,
                    nemo_file,
                    token=settings.HF_TOKEN,
                    cache_dir=self.model_root,
                )
                self.model = nemo_asr.models.ASRModel.restore_from(
                    restore_path=ckpt, map_location=torch.device(self.device)
                )

            if self.device == "cuda":
                self.model.cuda()
            self.model.freeze()
        except Exception as e:
            raise RuntimeError(f"NeMo load failed: {e}") from e

    def _transcribe_transformers(
        self, path: Path, language: str | None
    ) -> dict[str, Any]:
        """Execute transcription using the Transformers backend.

        Args:
            path: Path to the audio file.
            language: Language code.

        Returns:
            dict[str, Any]: Transcription result.
        """
        gen_kwargs = {"task": "transcribe"}
        if language:
            gen_kwargs["language"] = language

        if self.generation_config:
            gen_kwargs["generation_config"] = self.generation_config

        print(f"info: Generating with kwargs: {list(gen_kwargs.keys())}")

        # Run inference
        result = self.model(
            str(path), return_timestamps=True, generate_kwargs=gen_kwargs
        )

        text = result.get("text", "").strip()
        chunks = result.get("chunks", [])

        segments = []
        for c in chunks:
            # Handle potential None timestamps safely
            start = c["timestamp"][0] if c.get("timestamp") else 0.0
            end = c["timestamp"][1] if c.get("timestamp") else 0.0

            # If end is None (common in last chunk), approximate it
            if end is None:
                end = start + 2.0

            segments.append({"start": start, "end": end, "text": c["text"]})

        duration = segments[-1]["end"] if segments else 0.0

        return {
            "text": text,
            "segments": segments,
            "language": language or "unknown",
            "duration": duration,
        }

    def _transcribe_nemo(self, path: Path) -> dict[str, Any]:
        """Execute transcription using the NeMo backend.

        Args:
            path: Path to the audio file.

        Returns:
            dict[str, Any]: Transcription result.
        """
        # NeMo usually does batch inference and doesn't return easy timestamps
        # without complex forced alignment. Returning full text for now.
        files = [str(path)]
        text_result = self.model.transcribe(paths2audio_files=files, batch_size=1)[0]

        import librosa

        dur = librosa.get_duration(path=path)

        return {
            "text": str(text_result),
            "segments": [{"start": 0.0, "end": dur, "text": str(text_result)}],
            "language": "ta",
            "duration": dur,
        }

    def _find_project_root(self) -> Path:
        """Find the project root by looking for marker files.

        Returns:
            Path: The detected project root directory.
        """
        current = Path(__file__).resolve()
        for parent in current.parents:
            if (parent / ".git").exists() or (parent / "config.py").exists():
                return parent
        return current.parent.parent.parent


def main() -> None:
    """Test entry point for the module."""
    print("--- Audio Transcriber Test ---")

    test_file = Path("C:\\Users\\Gnana Prakash M\\Downloads\\dudeoutput.mp3")

    model_choice = "vasista22/whisper-tamil-large-v2"

    print(f"info: Initializing with model '{model_choice}'...")
    transcriber = AudioTranscriber(model_size=model_choice)

    if not test_file.exists():
        print(f"warning: Test file '{test_file}' not found. Exiting.")
        return

    try:
        result = transcriber.transcribe(test_file, language="ta")

        out_path = test_file.with_suffix(".txt")
        with open(out_path, "w", encoding="utf-8") as f:
            f.write(result["text"])

        print(f"\nSuccess! Transcript saved to {out_path}")
        print(f"Duration: {result['duration']:.2f}s")

        print("\nFirst 3 segments:")
        for seg in result["segments"][:3]:
            print(f"  [{seg['start']:.2f}-{seg['end']:.2f}] {seg['text']}")

    except Exception as e:
        print(f"\nError: {e}")


if __name__ == "__main__":
    main()
