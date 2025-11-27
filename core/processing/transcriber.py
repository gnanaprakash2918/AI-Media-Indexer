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

# Suppress external library warnings
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

        # Set up model storage in project_root/models
        if download_root:
            self.model_root = Path(download_root)
        else:
            self.model_root = self._find_project_root() / "models"
        self.model_root.mkdir(parents=True, exist_ok=True)

        self.model_id = model_size
        self.model: Any = None
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

        # Auto-detect language if not provided
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
        print("info: Starting transcription loop...")

        segments_list = []
        full_text_parts = []

        # Iterate generator to process segments and print progress
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
            # NVIDIA NeMo check
            if "rnnt" in lower or "nemo" in lower:
                return "nemo"
            return "transformers"  # Fallback for newer HF-compatible IndicConformers
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
            from transformers.pipelines import pipeline

            print(f"info: Pre-downloading '{model_id}' using HF Transfer (Fast)...")
            # Explicitly download first to ensure HF_TRANSFER is used and to avoid
            # pipeline timeouts.
            local_model_path = snapshot_download(
                repo_id=model_id,
                token=settings.HF_TOKEN,
                # Force download to a specific folder so we can pass that path to pipeline
                # if needed, but snapshot_download handles caching automatically.
            )
            print(f"info: Model downloaded to {local_model_path}")

            print("info: Initializing Transformers pipeline...")

            torch_dtype = torch.float16 if self.device == "cuda" else torch.float32

            self.model = pipeline(
                "automatic-speech-recognition",
                model=local_model_path,  # Load from local cache
                device=self.device,
                torch_dtype=torch_dtype,
                token=settings.HF_TOKEN,
                model_kwargs={"use_safetensors": False},
            )
        except ImportError as e:
            raise ImportError(
                "Please install 'transformers', 'accelerate', and 'hf_transfer' "
                "for this model."
            ) from e
        except Exception as e:
            raise RuntimeError(f"Failed to load Transformers model: {e}") from e

    def _load_nemo(self, model_id: str) -> None:
        """Load an NVIDIA NeMo model.

        Args:
            model_id: The model identifier.

        Raises:
            ImportError: If NeMo is not installed.
            RuntimeError: If model loading fails.
        """
        try:
            import nemo.collections.asr as nemo_asr
        except ImportError as e:
            raise ImportError(
                "NVIDIA NeMo is not installed. Please install it for AI4Bharat models."
            ) from e

        print("info: Fetching NeMo model file...")
        try:
            # Find .nemo file
            files = list_repo_files(model_id, token=settings.HF_TOKEN)
            nemo_file = next((f for f in files if f.endswith(".nemo")), None)

            if not nemo_file:
                # Fallback: try loading directly via class
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
                # Fix: Explicitly wrap device string in torch.device
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
        # Transformers specific kwargs
        gen_kwargs = {"task": "transcribe"}
        if language:
            gen_kwargs["language"] = language

        # Patch for some models missing timestamps
        result = self.model(
            str(path), return_timestamps=True, generate_kwargs=gen_kwargs
        )

        text = result.get("text", "").strip()
        chunks = result.get("chunks", [])
        segments = [
            {
                "start": c["timestamp"][0],
                "end": c["timestamp"][1] or 0.0,
                "text": c["text"],
            }
            for c in chunks
        ]
        return {
            "text": text,
            "segments": segments,
            "language": language or "unknown",
            "duration": segments[-1]["end"] if segments else 0.0,
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

        # Determine duration
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

    # User Configuration
    test_file = Path("test_audio.mp3")  # Replace with actual file if needed

    # FIX: Use 'vasista22/whisper-tamil-large-v2' (Robust Open SOTA)
    # This automatically routes to the 'transformers' engine.
    model_choice = "vasista22/whisper-tamil-large-v2"

    print(f"info: Initializing with model '{model_choice}' for better accuracy...")
    transcriber = AudioTranscriber(model_size=model_choice)

    if not test_file.exists():
        print(f"warning: Test file '{test_file}' not found. Exiting.")
        return

    try:
        result = transcriber.transcribe(test_file, language="ta")

        # FIX: Save output to the same directory as the audio file
        out_path = test_file.with_suffix(".txt")
        with open(out_path, "w", encoding="utf-8") as f:
            f.write(result["text"])

        print(f"\nSuccess! Transcript saved to {out_path}")
        print(f"Duration: {result['duration']:.2f}s")

    except KeyboardInterrupt:
        print("\nAborted by user.")
    except Exception as e:
        print(f"\nError: {e}")


if __name__ == "__main__":
    main()
