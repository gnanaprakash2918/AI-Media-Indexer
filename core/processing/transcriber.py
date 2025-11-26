"""Audio transcription utilities using a Tri-Hybrid Engine.

This module supports the full spectrum of modern ASR architectures:
1. NVIDIA NeMo: For AI4Bharat IndicConformer (Complex SOTA).
2. Transformers: For JiviAI/Vasista (Practical SOTA).
3. Faster-Whisper: For OpenAI models (Speed/General).

Usage:
    transcriber = AudioTranscriber()
    # Automatically picks best available engine (NeMo -> Transformers -> FW)
    result = transcriber.transcribe("path/to/file.mp3", language="ta")
"""

from __future__ import annotations

import os
import sys
import warnings
from pathlib import Path
from typing import Any, Literal

# --- 1. Suppress Python 3.12 SyntaxWarnings from Dependencies ---
warnings.filterwarnings("ignore", category=SyntaxWarning)
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

import librosa
import torch
from faster_whisper import WhisperModel
from huggingface_hub import hf_hub_download, list_repo_files, snapshot_download
from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor
from transformers.pipelines import pipeline

from config import settings
from core.schemas import TranscriptionResult

# Lazy import placeholder for NeMo
nemo_asr = None

os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "1"


class AudioTranscriber:
    """Tri-Hybrid Transcriber (NeMo, Transformers, Faster-Whisper)."""

    def __init__(
        self,
        model_size: str | None = None,
        compute_type: str | None = None,
        device: str | None = None,
        predownload: bool = True,
    ) -> None:
        self.explicit_model = model_size
        self.device = device or settings.WHISPER_DEVICE
        if not self.device:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"

        self.torch_device: torch.device = torch.device(self.device)

        self.compute_type = compute_type or settings.WHISPER_COMPUTE_TYPE
        if not self.compute_type:
            self.compute_type = "float16" if self.device == "cuda" else "float32"

        self.model: Any = None
        self.processor: Any = None
        self.model_id: str | None = None
        self.active_engine: Literal["nemo", "transformers", "faster-whisper"] = (
            "faster-whisper"
        )

        self.project_root = self._find_project_root()
        self.model_root_dir = self.project_root / "models"
        self.model_root_dir.mkdir(parents=True, exist_ok=True)
        self._predownload_enabled = predownload

        self._add_torch_libs_to_path(self.project_root)
        self._try_import_nemo()

        if self.explicit_model:
            self._load_with_fallback([self.explicit_model])

    def _try_import_nemo(self) -> None:
        """Lazily import NeMo to avoid crashes if not installed."""
        global nemo_asr
        try:
            import nemo.collections.asr as nemo_lib

            nemo_asr = nemo_lib
        except ImportError:
            pass

    def _determine_engine(self, model_id: str) -> str:
        """Heuristic to decide best engine for the model."""
        lower = model_id.lower()
        # Tier 0: AI4Bharat IndicConformer (Requires NeMo)
        if "indicconformer" in lower:
            return "nemo"
        # Tier 1/2: SOTA Whisper Fine-tunes (JiviAI/Vasista) -> Transformers
        if "jiviai" in lower or "audiox" in lower:
            return "transformers"
        if "vasista" in lower or "indicwhisper" in lower:
            return "transformers"
        # Tier 3: Standard OpenAI models -> Faster-Whisper (Speed)
        if "large-v3" in lower or "medium" in lower or "base" in lower:
            if "/" not in model_id:
                return "faster-whisper"
        return "faster-whisper"

    def _load_with_fallback(self, candidates: list[str]) -> None:
        """Try loading models from the list one by one until success."""
        errors: list[str] = []

        for model_id in candidates:
            engine = self._determine_engine(model_id)
            print(f"info: Attempting to load '{model_id}' (Engine: {engine})...")

            try:
                if engine == "nemo":
                    self._load_nemo_model(model_id)
                elif engine == "transformers":
                    self._load_transformers_pipeline(model_id)
                else:
                    self._load_faster_whisper_model(model_id)

                self.model_id = model_id
                self.active_engine = engine  # type: ignore
                print(f"success: Loaded '{model_id}'.")
                return

            except KeyboardInterrupt:
                print(f"\n\n[!] Download cancelled by user for '{model_id}'.")
                raise  # Stop the whole script if user cancels

            except Exception as e:
                msg = str(e)
                print(f"warning: Failed to load '{model_id}': {msg}")

                # --- Smart Hints for Common Errors ---
                if "401" in msg or "gated" in msg.lower():
                    print(f"\n[!] ACCESS DENIED: '{model_id}' is a Gated Model.")
                    print(
                        "    1. Accept license at: https://huggingface.co/" + model_id
                    )
                    print("    2. Set HF_TOKEN environment variable.\n")

                elif "nemo" in msg.lower() and engine == "nemo":
                    print(
                        "hint: NeMo toolkit missing. Run `pip install nemo_toolkit[asr]`"
                    )

                errors.append(f"{model_id}: {msg}")
                continue

        raise RuntimeError(f"All model candidates failed: {errors}")

    def _load_nemo_model(self, model_id: str) -> None:
        """Robustly load NeMo model by finding the .nemo file manually."""
        if nemo_asr is None:
            raise ImportError("NVIDIA NeMo is not installed.")

        try:
            # 1. Find the actual .nemo file in the repo (names vary)
            print(f"info: Searching for .nemo file in '{model_id}'...")
            files = list_repo_files(model_id, token=settings.HF_TOKEN)
            nemo_files = [f for f in files if f.endswith(".nemo")]

            if not nemo_files:
                raise FileNotFoundError(f"No .nemo file found in {model_id}")

            # Usually there's only one, take the first
            target_file = nemo_files[0]
            print(f"info: Found '{target_file}'. Downloading...")

            # 2. Download explicitly
            ckpt_path = hf_hub_download(
                repo_id=model_id, filename=target_file, token=settings.HF_TOKEN
            )

            # 3. Restore from local path
            print(f"info: Restoring NeMo model from: {ckpt_path}")
            self.model = nemo_asr.models.ASRModel.restore_from(
                restore_path=ckpt_path, map_location=torch.device(self.torch_device)
            )
        except Exception as e:
            print(
                f"warning: Manual .nemo load failed ({e}), trying standard fallback..."
            )
            # Fallback: standard from_pretrained (prone to config errors on Windows)
            self.model = nemo_asr.models.ASRModel.from_pretrained(model_name=model_id)

        if self.device == "cuda":
            self.model.cuda()
        self.model.freeze()

    def _load_transformers_pipeline(self, model_id: str) -> None:
        torch_dtype = torch.float16 if self.device == "cuda" else torch.float32

        # Load with explicit token and optimization
        model = AutoModelForSpeechSeq2Seq.from_pretrained(
            model_id,
            torch_dtype=torch_dtype,
            low_cpu_mem_usage=True,
            use_safetensors=True,
            token=settings.HF_TOKEN,
        )
        model.to(self.torch_device)

        processor = AutoProcessor.from_pretrained(model_id, token=settings.HF_TOKEN)

        self.model = pipeline(
            "automatic-speech-recognition",
            model=model,
            tokenizer=processor.tokenizer,
            feature_extractor=processor.feature_extractor,
            chunk_length_s=30,
            batch_size=16,
            torch_dtype=torch_dtype,
            device=self.device,
        )

    def _load_faster_whisper_model(self, model_id: str) -> None:
        if self._predownload_enabled:
            self._maybe_predownload(model_id)
        compute = "float16" if self.device == "cuda" else "int8"
        self.model = WhisperModel(
            model_id,
            device=self.device or "cpu",
            compute_type=compute,
            download_root=str(self.model_root_dir),
        )

    def transcribe(
        self,
        audio_path: str | Path,
        language: str | None = None,
        initial_prompt: str | None = None,
        beam_size: int = 5,
        vad_filter: bool = True,
    ) -> TranscriptionResult:
        path_obj = Path(audio_path)
        if not path_obj.exists():
            raise FileNotFoundError(f"Audio file not found: {path_obj}")

        candidates: list[str] = []
        if self.explicit_model:
            candidates = [self.explicit_model]
        else:
            lang_key = (language or "en").lower()
            candidates = settings.whisper_model_map.get(
                lang_key, [settings.WHISPER_MODEL]
            )

        if self.model_id not in candidates:
            print(f"info: Switching model for '{language}'. Candidates: {candidates}")
            self._load_with_fallback(candidates)

        # Tamil Prompt Injection
        if language == "ta" and not initial_prompt:
            initial_prompt = "வணக்கம், இது ஒரு தமிழ் உரையாடல் பதிவு."

        if self.active_engine == "nemo":
            return self._transcribe_nemo(path_obj)
        elif self.active_engine == "transformers":
            return self._transcribe_transformers(path_obj, language)
        else:
            return self._transcribe_fw(
                path_obj, language, initial_prompt, beam_size, vad_filter
            )

    def _transcribe_nemo(self, audio_path: Path) -> TranscriptionResult:
        try:
            # NeMo supports automatic batching for single files
            files = [str(audio_path)]
            # Force hybrid decoding if available (usually default)
            results = self.model.transcribe(paths2audio_files=files, batch_size=1)
            text = results[0] if isinstance(results, list) else str(results)

            return TranscriptionResult(
                text=text,
                segments=[{"start": 0.0, "end": 0.0, "text": text}],
                language="ta",
                language_probability=1.0,
                duration=librosa.get_duration(path=str(audio_path)),
            )
        except Exception as e:
            raise RuntimeError(f"NeMo Transcription failed: {e}")

    def _transcribe_transformers(
        self, audio_path: Path, language: str | None
    ) -> TranscriptionResult:
        generate_kwargs = {}
        if language:
            generate_kwargs["language"] = language

        result = self.model(
            str(audio_path), return_timestamps=True, generate_kwargs=generate_kwargs
        )

        text = result.get("text", "").strip()
        chunks = result.get("chunks", [])
        segments = [
            {
                "start": c.get("timestamp", (0, 0))[0],
                "end": c.get("timestamp", (0, 0))[1] or 0.0,
                "text": c.get("text", "").strip(),
            }
            for c in chunks
        ]

        duration = segments[-1]["end"] if segments else 0.0
        if duration == 0.0:
            duration = librosa.get_duration(path=str(audio_path))

        return TranscriptionResult(
            text=text,
            segments=segments,
            language=language or "unknown",
            language_probability=1.0,
            duration=duration,
        )

    def _transcribe_fw(
        self,
        audio_path: Path,
        language: str | None,
        prompt: str | None,
        beam: int,
        vad: bool,
    ) -> TranscriptionResult:
        if not self.model:
            raise RuntimeError("FW Model not loaded")
        segments_gen, info = self.model.transcribe(
            str(audio_path),
            beam_size=beam,
            vad_filter=vad,
            vad_parameters={"min_silence_duration_ms": 500} if vad else None,
            task="transcribe",
            language=language,
            initial_prompt=prompt,
        )
        segments = list(segments_gen)
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

    def _maybe_predownload(self, model_id: str) -> None:
        if "/" not in model_id:
            return
        try:
            snapshot_download(repo_id=model_id, token=settings.HF_TOKEN)
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
    print("Script Started")
    test_path = Path(r"C:\Users\Gnana Prakash M\Downloads\Programs\avengers.mp3")

    try:
        transcriber = AudioTranscriber(predownload=True)
        print("\n--- Testing SOTA Tamil Transcription ---")
        if test_path.exists():
            res = transcriber.transcribe(test_path, language="ta")
            print(f"Engine: {transcriber.active_engine.upper()}")
            print(f"Model:  {transcriber.model_id}")
            print(f"Text:   {res.text[:500]}...")

    except KeyboardInterrupt:
        print("\n\n[!] Script Interrupted by User.")
    except Exception as exc:
        print(f"error: Test run failed: {exc}")


if __name__ == "__main__":
    main()
