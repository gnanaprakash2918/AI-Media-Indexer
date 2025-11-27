"""Audio transcription utilities using a Tri-Hybrid Engine.

This module supports the full spectrum of modern ASR architectures:
1. NVIDIA NeMo: For AI4Bharat IndicConformer (Complex SOTA).
2. Transformers: For JiviAI/Vasista (Practical SOTA).
3. Faster-Whisper: For OpenAI models (Speed/General).
"""

from __future__ import annotations

import os
import sys
import warnings
from pathlib import Path
from typing import Any, Literal, cast

# --- 1. Suppress Python 3.12 SyntaxWarnings from Dependencies ---
warnings.filterwarnings("ignore", category=SyntaxWarning)
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

import librosa
import torch
import torchaudio
from faster_whisper import WhisperModel
from huggingface_hub import hf_hub_download, list_repo_files, snapshot_download
from transformers import (
    AutoModel,
    AutoModelForSpeechSeq2Seq,
    AutoProcessor,
)
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
        enable_language_detection: bool = True,
        preferred_tamil_provider: Literal[
            "ai4bharat", "whisper", "transformers", "auto"
        ] = "ai4bharat",
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

        self.enable_language_detection = enable_language_detection
        self.preferred_tamil_provider = preferred_tamil_provider

        self._lang_detect_model: WhisperModel | None = None

        self.project_root = self._find_project_root()
        self.model_root_dir = self.project_root / "models"
        self.model_root_dir.mkdir(parents=True, exist_ok=True)
        self._predownload_enabled = predownload

        self._add_torch_libs_to_path(self.project_root)
        self._try_import_nemo()

        # Set HF Cache paths
        os.environ["HF_HOME"] = str(self.model_root_dir)
        os.environ["XDG_CACHE_HOME"] = str(self.model_root_dir)
        os.environ["TRANSFORMERS_CACHE"] = str(self.model_root_dir)
        os.environ["HF_HUB_CACHE"] = str(self.model_root_dir)

        if self.explicit_model:
            self._load_with_fallback([self.explicit_model])

    def _try_import_nemo(self) -> None:
        global nemo_asr
        try:
            import nemo.collections.asr as nemo_lib

            nemo_asr = nemo_lib
        except ImportError:
            nemo_asr = None

    def _determine_engine(self, model_id: str) -> str:
        lower = model_id.lower()
        if "indicconformer_stt_ta_hybrid_ctc_rnnt_large" in lower:
            raise RuntimeError("Old NeMo-based IndicConformer Tamil model is disabled.")
        if "indic-conformer-600m-multilingual" in lower:
            return "transformers"
        if "jiviai" in lower or "audiox" in lower:
            return "transformers"
        if "vasista" in lower or "indicwhisper" in lower:
            return "transformers"
        if "large-v3" in lower or "medium" in lower or "base" in lower:
            if "/" not in model_id:
                return "faster-whisper"
        return "faster-whisper"

    def _load_with_fallback(self, candidates: list[str]) -> None:
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
                self.active_engine = cast(
                    Literal["nemo", "transformers", "faster-whisper"], engine
                )
                print(f"success: Loaded '{model_id}'.")
                return

            except KeyboardInterrupt:
                print(f"\n\n[!] Download cancelled by user for '{model_id}'.")
                raise
            except Exception as e:
                msg = str(e)
                print(f"warning: Failed to load '{model_id}': {msg}")
                errors.append(f"{model_id}: {msg}")
                continue

        raise RuntimeError(f"All model candidates failed: {errors}")

    def _load_nemo_model(self, model_id: str) -> None:
        if nemo_asr is None:
            raise ImportError("NVIDIA NeMo is not installed.")
        try:
            print(f"info: Searching for .nemo file in '{model_id}'...")
            files = list_repo_files(model_id, token=settings.HF_TOKEN)
            nemo_files = [f for f in files if f.endswith(".nemo")]

            if not nemo_files:
                raise FileNotFoundError(f"No .nemo file found in {model_id}")

            target_file = nemo_files[0]
            print(f"info: Found '{target_file}'. Downloading...")

            ckpt_path = hf_hub_download(
                repo_id=model_id, filename=target_file, token=settings.HF_TOKEN
            )
            print(f"info: Restoring NeMo model from: {ckpt_path}")
            self.model = nemo_asr.models.ASRModel.restore_from(
                restore_path=ckpt_path, map_location=torch.device(self.torch_device)
            )
        except Exception as e:
            print(
                f"warning: Manual .nemo load failed ({e}), trying standard fallback..."
            )
            self.model = nemo_asr.models.ASRModel.from_pretrained(model_name=model_id)

        if self.device == "cuda":
            self.model.cuda()
        self.model.freeze()

    def _load_transformers_pipeline(self, model_id: str) -> None:
        """Load a Transformers-based ASR model or pipeline."""
        if "indic-conformer-600m-multilingual" in model_id:
            torch_dtype = torch.float16 if self.device == "cuda" else torch.float32
            self.model = AutoModel.from_pretrained(
                model_id,
                trust_remote_code=True,
                torch_dtype=torch_dtype,
                token=settings.HF_TOKEN,
            ).to(self.torch_device)
            self.processor = None
            return

        torch_dtype = torch.float16 if self.device == "cuda" else torch.float32

        model = AutoModelForSpeechSeq2Seq.from_pretrained(
            model_id,
            torch_dtype=torch_dtype,
            low_cpu_mem_usage=True,
            use_safetensors=True,
            token=settings.HF_TOKEN,
        )
        model.to(self.torch_device)

        # --- FIX FOR TIMESTAMP ERROR (Vasista/Whisper specific) ---
        if "whisper" in model_id.lower():
            try:
                # Force update the generation config
                # 50363 is the <|notimestamps|> token for Whisper V2 models
                model.generation_config.no_timestamps_token_id = 50363
                model.config.no_timestamps_token_id = 50363

                # Clear forced settings that might conflict with pipeline
                if hasattr(model.generation_config, "forced_decoder_ids"):
                    model.generation_config.forced_decoder_ids = None

            except Exception as e:
                print(f"warning: Could not patch generation config: {e}")
                pass
        # -------------------------------

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
        self.processor = processor

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
        prefer_ai4bharat: bool | None = None,
    ) -> TranscriptionResult:
        path_obj = Path(audio_path)
        if not path_obj.exists():
            raise FileNotFoundError(f"Audio file not found: {path_obj}")

        if language is not None:
            normalized_lang = self._normalize_language_code(language)
        else:
            auto_lang = self._detect_language_from_audio(path_obj)
            normalized_lang = self._normalize_language_code(auto_lang)

        lang_key = (normalized_lang or "en").lower()

        effective_prefer_ai4bharat = (
            prefer_ai4bharat
            if prefer_ai4bharat is not None
            else self.preferred_tamil_provider == "ai4bharat"
        )

        candidates: list[str] = []
        if self.explicit_model:
            candidates = [self.explicit_model]
        else:
            candidates = settings.whisper_model_map.get(
                lang_key, [settings.WHISPER_MODEL]
            )

        if lang_key == "ta":
            candidates = self._reorder_tamil_candidates(
                candidates, effective_prefer_ai4bharat
            )

        if self.model_id not in candidates:
            print(
                f"info: Switching model for '{normalized_lang}'. Candidates: {candidates}"
            )
            self._load_with_fallback(candidates)

        if normalized_lang == "ta" and not initial_prompt:
            initial_prompt = "வணக்கம், இது ஒரு தமிழ் உரையாடல் பதிவு."

        if self.active_engine == "nemo":
            return self._transcribe_nemo(path_obj)
        if self.active_engine == "transformers":
            if self.model_id and "indic-conformer-600m-multilingual" in self.model_id:
                return self._transcribe_indic_conformer(path_obj, normalized_lang)
            return self._transcribe_transformers(path_obj, normalized_lang)
        return self._transcribe_fw(
            path_obj, normalized_lang, initial_prompt, beam_size, vad_filter
        )

    def _transcribe_nemo(self, audio_path: Path) -> TranscriptionResult:
        try:
            files = [str(audio_path)]
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

    def _transcribe_indic_conformer(
        self, audio_path: Path, language: str | None
    ) -> TranscriptionResult:
        if self.model is None:
            raise RuntimeError("Indic-Conformer model not loaded")

        wav, sr = torchaudio.load(str(audio_path))
        if wav.dtype != torch.float32:
            wav = wav.to(torch.float32)

        target_sr = 16000
        if sr != target_sr:
            resampler = torchaudio.transforms.Resample(orig_freq=sr, new_freq=target_sr)
            wav = resampler(wav)

        if wav.ndim == 1:
            wav = wav.unsqueeze(0)
        elif wav.ndim == 2 and wav.size(0) > 1:
            wav = wav.mean(dim=0, keepdim=True)

        max_val = float(wav.abs().max()) if wav.numel() > 0 else 0.0
        if max_val > 1.0:
            wav = wav / max_val

        lang_code = self._normalize_language_code(language) or "ta"

        chunk_duration_s = 30
        overlap_s = 1
        hop_s = max(1, chunk_duration_s - overlap_s)
        chunk_samples = int(chunk_duration_s * target_sr)
        hop_samples = int(hop_s * target_sr)
        total_samples = wav.shape[1]

        full_text_parts: list[str] = []
        segments: list[dict[str, Any]] = []

        print(
            f"info: Processing {total_samples / target_sr:.2f}s audio in {chunk_duration_s}s chunks..."
        )

        for start in range(0, total_samples, hop_samples):
            end = min(start + chunk_samples, total_samples)
            chunk_wav = wav[:, start:end]
            if chunk_wav.numel() == 0:
                continue

            chunk_wav = chunk_wav.to(self.torch_device)
            start_time = start / target_sr
            end_time = end / target_sr

            try:
                with torch.no_grad():
                    chunk_text = self.model(chunk_wav, lang_code, "rnnt")

                if isinstance(chunk_text, (list, tuple)) and chunk_text:
                    text_value = str(chunk_text[0]).strip()
                else:
                    text_value = str(chunk_text).strip()

                if not text_value:
                    continue

                if full_text_parts and text_value.startswith(
                    full_text_parts[-1].split()[-3:][0]
                    if full_text_parts[-1].split()
                    else ""
                ):
                    pass

                print(f"  [{start_time:.1f}s -> {end_time:.1f}s]: {text_value[:50]}...")
                full_text_parts.append(text_value)
                segments.append(
                    {"start": start_time, "end": end_time, "text": text_value}
                )

            except Exception as e:
                print(f"warning: Chunk failed at {start_time}s: {e}")
                continue

            if end >= total_samples:
                break

        final_text = " ".join(full_text_parts).strip()
        try:
            duration = librosa.get_duration(path=str(audio_path))
        except Exception:
            duration = float(total_samples) / target_sr

        return TranscriptionResult(
            text=final_text,
            segments=segments,
            language=lang_code,
            language_probability=1.0,
            duration=duration,
        )

    def _transcribe_transformers(
        self, audio_path: Path, language: str | None
    ) -> TranscriptionResult:
        # --- CONFIGURING GENERATION KWARGS ---
        generate_kwargs: dict[str, Any] = {
            "task": "transcribe",
            "language": "tamil" if language == "ta" else language,
            # CRITICAL FIX: Explicitly pass the missing token ID here
            # This overrides whatever is missing in the config
            "no_timestamps_token_id": 50363,
        }

        result = self.model(
            str(audio_path), return_timestamps=True, generate_kwargs=generate_kwargs
        )

        text = result.get("text", "").strip()
        chunks = result.get("chunks", [])
        segments = [
            {
                "start": c.get("timestamp", (0.0, 0.0))[0],
                "end": c.get("timestamp", (0.0, 0.0))[1] or 0.0,
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

        def _segments_to_list(segments_gen):
            segs = list(segments_gen)
            out = []
            for s in segs:
                if hasattr(s, "text"):
                    text = getattr(s, "text", "")
                    start = getattr(s, "start", 0.0)
                    end = getattr(s, "end", 0.0)
                elif isinstance(s, dict):
                    text = s.get("text", "")
                    start = s.get("start", 0.0)
                    end = s.get("end", 0.0)
                else:
                    text = str(s)
                    start = 0.0
                    end = 0.0
                out.append(
                    {
                        "start": float(start or 0.0),
                        "end": float(end or 0.0),
                        "text": str(text or "").strip(),
                    }
                )
            return out

        segments_gen, info = self.model.transcribe(
            str(audio_path),
            beam_size=beam,
            vad_filter=vad,
            vad_parameters={"min_silence_duration_ms": 500} if vad else None,
            task="transcribe",
            language=language,
            initial_prompt=prompt,
        )
        segments_list = _segments_to_list(segments_gen)

        if not segments_list and vad:
            try:
                segments_gen2, info = self.model.transcribe(
                    str(audio_path),
                    beam_size=max(1, beam),
                    vad_filter=False,
                    task="transcribe",
                    language=language,
                    initial_prompt=prompt,
                )
                segments_list = _segments_to_list(segments_gen2)
            except Exception:
                pass

        full_text = " ".join([s["text"] for s in segments_list if s["text"]]).strip()
        lang = getattr(info, "language", language or "")
        lang_prob = getattr(info, "language_probability", 0.0)
        duration = getattr(info, "duration", 0.0)
        if not duration:
            try:
                duration = librosa.get_duration(path=str(audio_path))
            except Exception:
                duration = 0.0

        return TranscriptionResult(
            text=full_text,
            segments=segments_list,
            language=lang,
            language_probability=lang_prob,
            duration=duration,
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

    def _normalize_language_code(self, language: str | None) -> str | None:
        if language is None:
            return None
        code = language.strip().lower()
        alias_map = getattr(settings, "LANGUAGE_CODE_ALIASES", None)
        if isinstance(alias_map, dict):
            mapped = alias_map.get(code)
            if isinstance(mapped, str) and mapped:
                return mapped
        if len(code) == 2:
            return code
        if "-" in code:
            parts = code.split("-")
            if parts and len(parts[0]) == 2:
                return parts[0]
        return code

    def _reorder_tamil_candidates(
        self, candidates: list[str], prefer_ai4bharat: bool
    ) -> list[str]:
        if not candidates:
            return candidates
        ai4bharat_models: list[str] = []
        whisper_models: list[str] = []
        transformer_others: list[str] = []
        remaining: list[str] = []

        for m in candidates:
            lower = m.lower()
            if "ai4bharat" in lower:
                ai4bharat_models.append(m)
            elif "whisper" in lower or "large-v3" in lower:
                whisper_models.append(m)
            elif "jiviai" in lower or "audiox" in lower or "vasista" in lower:
                transformer_others.append(m)
            else:
                remaining.append(m)

        if prefer_ai4bharat:
            ordered = ai4bharat_models + transformer_others + whisper_models + remaining
        else:
            ordered = transformer_others + whisper_models + ai4bharat_models + remaining
        if not ordered:
            return candidates
        return ordered

    def _detect_language_from_audio(self, audio_path: Path) -> str | None:
        if not self.enable_language_detection:
            return None
        try:
            if self._lang_detect_model is None:
                model_id = getattr(settings, "LANGUAGE_DETECT_MODEL", "tiny")
                self._lang_detect_model = WhisperModel(
                    model_id,
                    device="cpu",
                    compute_type="int8",
                    download_root=str(self.model_root_dir),
                )
            segments_gen, info = self._lang_detect_model.transcribe(
                str(audio_path),
                beam_size=1,
                vad_filter=False,
                task="transcribe",
            )
            _ = list(segments_gen)
            lang = getattr(info, "language", None)
            if isinstance(lang, str) and lang:
                return lang
        except Exception:
            return None
        return None


def main() -> None:
    print("Script Started")
    test_path = Path(r"C:\Users\Gnana Prakash M\Downloads\Programs\avengers.mp3")

    try:
        # CHANGE: Use 'large-v3' (OpenAI's latest) with the 'auto' provider.
        # This routes the logic to the 'faster-whisper' engine, which is bug-free for timestamps.
        transcriber = AudioTranscriber(
            model_size="large-v3",
            predownload=True,
            preferred_tamil_provider="auto",  # Allows switching to faster-whisper
        )

        print("\n--- Testing Robust Tamil Transcription (Faster-Whisper) ---")
        if test_path.exists():
            # vad_filter=True helps skip the silence/music in your 2min clip
            res = transcriber.transcribe(
                test_path, language="ta", vad_filter=True, beam_size=5
            )

            print(f"Engine: {transcriber.active_engine.upper()}")
            print(f"Model:  {transcriber.model_id}")

            output_file = test_path.with_suffix(".txt")
            with open(output_file, "w", encoding="utf-8") as f:
                f.write(res.text)

            print(f"\n[+] Saved full transcript to: {output_file}")
            print(f"[+] Total Characters: {len(res.text)}")
            print(f"[+] Total Segments (Frames): {len(res.segments)}")

            if res.segments:
                print("\n--- Frame Preview (Timestamps) ---")
                for seg in res.segments:
                    # Print all segments since it's only 2 mins
                    print(f"[{seg['start']:.2f}s -> {seg['end']:.2f}s]: {seg['text']}")

    except KeyboardInterrupt:
        print("\n\n[!] Script Interrupted by User.")
    except Exception as exc:
        print(f"error: Test run failed: {exc}")


if __name__ == "__main__":
    main()
