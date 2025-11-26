"""Audio transcription utilities supporting Hybrid Engines (Whisper & Wav2Vec).

This module provides a unified wrapper that supports:
1. Transformers (Wav2Vec2): For AI4Bharat/IndicWav2Vec models (Tier 1 Performance).
2. Faster-Whisper (CTranslate2): For JiviAI, Vasista, OpenAI models.

It implements a "Best -> Worst" fallback strategy based on performance.

Usage:
    transcriber = AudioTranscriber()
    result = transcriber.transcribe("path/to/file.mp3", language="ta")
"""

from __future__ import annotations

import os
import sys
import warnings
from pathlib import Path
from typing import Any

import librosa
import torch
from faster_whisper import WhisperModel
from huggingface_hub import snapshot_download
from transformers import AutoModelForCTC, AutoProcessor

from config import settings
from core.schemas import TranscriptionResult

# Filter warnings
warnings.filterwarnings("ignore", message=".*pkg_resources is deprecated.*")
warnings.filterwarnings("ignore", category=UserWarning, module="ctranslate2")
warnings.filterwarnings("ignore", category=UserWarning, module="transformers")

os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "1"


class AudioTranscriber:
    """Hybrid Transcriber supporting Faster-Whisper and Wav2Vec2 architectures.

    Attributes:
        active_engine: "whisper" or "wav2vec".
        model: The loaded model object (WhisperModel or HF Model).
        processor: The HF Processor (only for Wav2Vec).
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
            model_size: Explicit model ID. If None, it is chosen dynamically based
                on the language during `transcribe()`.
            compute_type: Precision override (e.g., "float16", "int8").
            device: "cuda" or "cpu".
            predownload: Enable HF pre-downloading.
        """
        self.explicit_model = model_size
        self.device = device or settings.WHISPER_DEVICE
        if not self.device:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"

        self.compute_type = compute_type or settings.WHISPER_COMPUTE_TYPE
        if not self.compute_type:
            self.compute_type = "float16" if self.device == "cuda" else "int8"

        self.model: Any = None
        # For Wav2Vec
        self.processor: Any = None

        self.model_id: str | None = None

        # "whisper" or "wav2vec"
        self.active_engine: str = "whisper"

        self.project_root = self._find_project_root()
        self.model_root_dir = self.project_root / "models"
        self.model_root_dir.mkdir(parents=True, exist_ok=True)
        self._predownload_enabled = predownload

        self._add_torch_libs_to_path(self.project_root)

        if self.explicit_model:
            self._load_with_fallback([self.explicit_model])

    def _determine_engine(self, model_id: str) -> str:
        """Heuristic to decide if model is Whisper or Wav2Vec."""
        lower = model_id.lower()
        if "wav2vec" in lower or "indicconformer" in lower:
            return "wav2vec"

        # JiviAI audioX is Whisper based
        if "audiox" in lower or "whisper" in lower or "distil" in lower:
            return "whisper"
        # Default to Whisper for unknown names
        return "whisper"

    def _load_with_fallback(self, candidates: list[str]) -> None:
        """Try loading models from the list one by one until success."""
        errors: list[str] = []
        for model_id in candidates:
            engine = self._determine_engine(model_id)
            print(f"info: Attempting to load '{model_id}' (Engine: {engine})...")

            try:
                if engine == "whisper":
                    self._load_whisper_model(model_id)
                else:
                    self._load_wav2vec_model(model_id)

                self.model_id = model_id
                self.active_engine = engine
                print(f"success: Loaded '{model_id}'.")
                return
            except Exception as e:
                print(f"warning: Failed to load '{model_id}': {e}")
                errors.append(f"{model_id}: {str(e)}")
                continue

        raise RuntimeError(f"All model candidates failed to load: {errors}")

    def _load_whisper_model(self, model_id: str) -> None:
        """Load Faster-Whisper model."""
        if self._predownload_enabled:
            self._maybe_predownload(model_id)

        device = (
            self.device
            if self.device is not None
            else ("cuda" if torch.cuda.is_available() else "cpu")
        )

        compute_type = (
            self.compute_type
            if self.compute_type is not None
            else ("float16" if device == "cuda" else "int8")
        )

        self.model = WhisperModel(
            model_id,
            device=device,
            compute_type=compute_type,
            download_root=str(self.model_root_dir),
        )
        self.processor = None

    def _load_wav2vec_model(self, model_id: str) -> None:
        """Load Transformers Wav2Vec2 model."""
        print(f"info: Loading Transformers pipeline for {model_id}...")
        try:
            self.processor = AutoProcessor.from_pretrained(
                model_id, token=settings.HF_TOKEN
            )
            self.model = AutoModelForCTC.from_pretrained(
                model_id, token=settings.HF_TOKEN
            )
            if self.device == "cuda":
                self.model.to("cuda")
        except OSError as e:
            if "gated" in str(e).lower() or "token" in str(e).lower():
                raise PermissionError(f"Access denied (Gated Model?): {e}") from e
            raise e

    def transcribe(
        self,
        audio_path: str | Path,
        language: str | None = None,
        initial_prompt: str | None = None,
        beam_size: int = 5,
        vad_filter: bool = True,
    ) -> TranscriptionResult:
        """Transcribe audio using the best available model for the language.

        Args:
            audio_path: Path to input audio.
            language: ISO language code (e.g., 'ta').
            initial_prompt: Optional prompt (Whisper only).
            beam_size: Beam search width (Whisper only).
            vad_filter: Enable VAD (Whisper only).

        Returns:
            TranscriptionResult object.
        """
        path_obj = Path(audio_path)
        if not path_obj.exists():
            raise FileNotFoundError(f"Audio file not found: {path_obj}")

        # 1. Determine Candidates
        candidates: list[str] = []
        if self.explicit_model:
            candidates = [self.explicit_model]
        else:
            lang_key = (language or "en").lower()
            candidates = settings.whisper_model_map.get(
                lang_key, [settings.WHISPER_MODEL]
            )

        # 2. Check if current loaded model is usable
        # If current model is not in candidates, we must reload.
        if self.model_id not in candidates:
            print(
                f"info: Switching model for language '{language}'. "
                f"Candidates: {candidates}"
            )
            self._load_with_fallback(candidates)

        # 3. Apply Tamil Prompt Fix (Whisper only)
        if self.active_engine == "whisper" and language == "ta" and not initial_prompt:
            initial_prompt = "வணக்கம், இது ஒரு தமிழ் உரையாடல் பதிவு."

        # 4. Dispatch
        if self.active_engine == "whisper":
            return self._transcribe_whisper(
                path_obj, language, initial_prompt, beam_size, vad_filter
            )
        else:
            return self._transcribe_wav2vec(path_obj, language)

    def _transcribe_whisper(
        self,
        audio_path: Path,
        language: str | None,
        prompt: str | None,
        beam: int,
        vad: bool,
    ) -> TranscriptionResult:
        """Internal handler for Faster-Whisper."""
        if not isinstance(self.model, WhisperModel):
            raise RuntimeError("Engine mismatch: Expected WhisperModel")

        segments_generator, info = self.model.transcribe(
            str(audio_path),
            beam_size=beam,
            vad_filter=vad,
            vad_parameters={"min_silence_duration_ms": 500} if vad else None,
            task="transcribe",
            language=language,
            initial_prompt=prompt,
        )

        segments = list(segments_generator)
        full_text = "".join([s.text for s in segments]).strip()

        return TranscriptionResult(
            text=full_text,
            segments=[
                {"start": s.start, "end": s.end, "text": s.text.strip()}
                for s in segments
            ],
            language=getattr(info, "language", language or ""),
            language_probability=getattr(info, "language_probability", 0.0),
            duration=getattr(info, "duration", 0.0),
        )

    def _transcribe_wav2vec(
        self, audio_path: Path, language: str | None
    ) -> TranscriptionResult:
        """Internal handler for Wav2Vec2/IndicConformer."""
        if self.processor is None or self.model is None:
            raise RuntimeError("Engine mismatch: Transformers model not loaded")

        # Wav2Vec requires 16kHz
        try:
            audio_input, sr = librosa.load(str(audio_path), sr=16000)
        except Exception as e:
            raise RuntimeError(f"Librosa load failed: {e}") from e

        duration = librosa.get_duration(y=audio_input, sr=16000)

        # Tokenize
        inputs = self.processor(audio_input, sampling_rate=16000, return_tensors="pt")
        if self.device == "cuda":
            inputs = inputs.to("cuda")

        # Inference
        with torch.no_grad():
            logits = self.model(**inputs).logits

        # Decode
        predicted_ids = torch.argmax(logits, dim=-1)
        # skip_special_tokens=True removes padding/CTC tokens
        transcription_list = self.processor.batch_decode(
            predicted_ids, skip_special_tokens=True
        )
        transcription = str(transcription_list[0]) if transcription_list else ""

        # Wav2Vec doesn't provide segments naturally.
        # We create a single segment for schema compatibility.
        return TranscriptionResult(
            text=transcription,
            segments=[{"start": 0.0, "end": duration, "text": transcription}],
            language=language or "ta",
            language_probability=1.0,
            duration=duration,
        )

    def _maybe_predownload(self, model_id: str) -> None:
        """Attempt to pre-download HF model."""
        if "/" not in model_id:
            return
        try:
            snapshot_download(
                repo_id=model_id,
                allow_patterns=[
                    "config.json",
                    "*.bin",
                    "*.pt",
                    "*.safetensors",
                    "tokenizer.json",
                    "vocab.json",
                ],
                token=settings.HF_TOKEN,
            )
        except Exception:
            pass

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
            pass

    def _find_project_root(self) -> Path:
        current = Path(__file__).resolve()
        for parent in current.parents:
            if any(
                (parent / marker).exists()
                for marker in ["pyproject.toml", ".venv", ".git"]
            ):
                return parent
        return current.parent.parent.parent


def main() -> None:
    """Small manual test harness."""
    print("Script Started")
    # Test path
    test_path = Path(r"C:\Users\Gnana Prakash M\Downloads\Programs\endgame-english.mp3")

    try:
        # Initialize
        transcriber = AudioTranscriber(predownload=True)

        print("\n--- Testing Tamil Transcription (Hybrid Fallback) ---")
        if test_path.exists():
            # Should try AI4Bharat (Wav2Vec) first, then JiviAI, then Vasista
            res = transcriber.transcribe(
                test_path,
                language="ta",
                beam_size=5,
            )
            print(f"Result (First 10000 chars): {res.text[:10000]}...")
            print(f"Active Engine used: {transcriber.active_engine}")
            print(f"Active Model: {transcriber.model_id}")

            print("\nSegments:")
            for seg in res.segments or []:
                start = seg.get("start", 0.0)
                end = seg.get("end", 0.0)
                txt = seg.get("text", "").strip()
                print(f" - [{start:.2f}s -> {end:.2f}s] {txt}")

    except Exception as exc:
        print(f"error: Test run failed: {exc}")


if __name__ == "__main__":
    main()
