"""Robust Audio Transcriber using Faster-Whisper and AI4Bharat NeMo.

Features:
- Docker Ready: Optimized for containerized execution with Windows path mapping.
- Hardware Agnostic: Auto-switch between GPU (CUDA) and CPU.
- Hybrid Engine: NeMo (Tamil) + Faster-Whisper (Global).
- Smart Caching: Project Local > Global Cache > Download.
- Subtitle Probing: Checks for embedded streams or sidecar files before transcribing.
- Tanglish Fixed: Cleanups specific to Tamil text.
"""

print("[INFO] LOADING TRANSCRIBER (v5 - SMART CONFIG SEARCH)...")

import gc
import os
import re
import shutil
import subprocess
import sys
import tarfile
import unicodedata
import warnings
from pathlib import Path
from tempfile import NamedTemporaryFile
from typing import Any

import ctranslate2.converters
import torch
import torchaudio
import yaml
from faster_whisper import BatchedInferencePipeline, WhisperModel
from huggingface_hub import hf_hub_download, snapshot_download

# Required for the NeMo Docker compatibility patch
from omegaconf import OmegaConf

from config import settings

warnings.filterwarnings("ignore")

# Optimization Flags
os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "1"
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"


class AudioTranscriber:
    """Main transcription class handling ASR lifecycle for Whisper and NeMo."""

    def __init__(self) -> None:
        """Initialize the AudioTranscriber instance."""
        self._model: WhisperModel | None = None
        self._batched_model: BatchedInferencePipeline | None = None
        self._nemo_model: Any | None = None
        self._current_model_id: str | None = None

        self.device = settings.device
        self.compute_type = settings.compute_type

        print(
            f"[INFO] Initialized AudioTranscriber\n"
            f"       Device: {self.device}\n"
            f"       Compute: {self.compute_type}"
        )

    def _resolve_path(self, raw_path: str) -> Path:
        """Converts Windows paths to Docker internal paths if running in Docker."""
        clean_path = raw_path.strip().strip('"').strip("'")
        match = re.match(r"^([a-zA-Z]):[\\/](.*)", clean_path)
        if match:
            drive_letter = match.group(1).lower()
            rest_of_path = match.group(2).replace("\\", "/")
            docker_path = Path(f"/mnt/{drive_letter}/{rest_of_path}")
            if not docker_path.exists():
                print(f"[WARN] Converted path does not exist: {docker_path}")
                return Path(clean_path).resolve()
            return docker_path
        return Path(clean_path).resolve()

    def _get_ffmpeg_cmd(self) -> str:
        cmd = shutil.which("ffmpeg")
        if not cmd:
            raise RuntimeError("FFmpeg not found. Please install it to system PATH.")
        return cmd

    def _find_existing_subtitles(
        self,
        input_path: Path,
        output_path: Path,
        user_sub_path: Path | None,
        language: str,
    ) -> bool:
        print("[INFO] Probing for existing subtitles...")
        if user_sub_path and user_sub_path.exists():
            print(f"[SUCCESS] Using provided subtitle: {user_sub_path}")
            shutil.copy(user_sub_path, output_path)
            return True

        for sidecar in [
            input_path.with_suffix(f".{language}.srt"),
            input_path.with_suffix(".srt"),
        ]:
            if sidecar.exists() and sidecar != output_path:
                print(f"[SUCCESS] Found sidecar: {sidecar}")
                shutil.copy(sidecar, output_path)
                return True

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

    def _prepare_16k_mono_wav(self, in_path: Path) -> Path:
        waveform, sr = torchaudio.load(str(in_path))
        if waveform.shape[0] > 1:
            waveform = waveform.mean(dim=0, keepdim=True)
        if sr != 16000:
            waveform = torchaudio.functional.resample(waveform, sr, 16000)
        tmp = NamedTemporaryFile(suffix=".wav", delete=False)
        tmp_path = Path(tmp.name)
        tmp.close()
        torchaudio.save(str(tmp_path), waveform, 16000)
        return tmp_path

    def _get_audio_duration(self, path: Path) -> float:
        info = torchaudio.info(str(path))
        return info.num_frames / info.sample_rate

    def _clean_tamil_text(self, text: str) -> str:
        result = []
        prev_is_tamil_base = False
        for ch in text:
            cat = unicodedata.category(ch)
            if cat.startswith("M"):
                if not prev_is_tamil_base:
                    continue
            else:
                if "\u0b80" <= ch <= "\u0bff":
                    prev_is_tamil_base = True
                else:
                    prev_is_tamil_base = False
            result.append(ch)
        cleaned = "".join(result)
        return re.sub(r"\s+", " ", cleaned).strip()

    def _slice_audio(self, input_path: Path, start: float, end: float | None) -> Path:
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
        subprocess.run(cmd, check=True, stderr=subprocess.DEVNULL)
        return output_slice

    def _split_long_chunks(
        self, chunks: list[dict[str, Any]], max_segment_s: float = 8.0
    ) -> list[dict[str, Any]]:
        out: list[dict[str, Any]] = []
        for ch in chunks:
            text = (ch.get("text") or "").strip()
            timestamp = ch.get("timestamp")
            if not text or not timestamp:
                continue
            try:
                start, end = timestamp
            except Exception:
                out.append(ch)
                continue
            if start is None:
                continue
            if end is None:
                end = start + 2.0
            start, end = float(start), float(end)
            duration = end - start

            if duration <= max_segment_s:
                out.append(ch)
                continue

            n = int((duration + max_segment_s - 1e-9) // max_segment_s) + 1
            n = max(1, n)
            words = text.split()
            if not words:
                out.append(ch)
                continue

            per = max(1, len(words) // n)
            ptr = 0
            for i in range(n):
                sub_words = words[ptr : ptr + per]
                ptr += per
                if not sub_words:
                    continue
                sub_text = " ".join(sub_words)
                sub_start = start + i * (duration / n)
                sub_end = start + (i + 1) * (duration / n)
                out.append({"text": sub_text, "timestamp": (sub_start, sub_end)})
            if ptr < len(words) and out:
                out[-1]["text"] = out[-1]["text"] + " " + " ".join(words[ptr:])
        return out

    def _find_model_config_root(self, conf: dict) -> dict:
        """Recursively search for the dict containing 'encoder' key."""
        if "encoder" in conf:
            return conf

        for k, v in conf.items():
            if isinstance(v, dict):
                res = self._find_model_config_root(v)
                if res is not None:
                    return res
        return None

    def _load_nemo_model(self) -> None:
        if self._nemo_model:
            return
        print(f"[INFO] Loading AI4Bharat IndicConformer (NeMo) on {self.device}...")
        try:
            from nemo.collections.asr.models import EncDecCTCModelBPE
        except ImportError as e:
            raise ImportError("NeMo not installed.") from e

        if self._model:
            del self._model
            del self._batched_model
            self._model, self._batched_model = None, None
            gc.collect()
            if self.device == "cuda":
                torch.cuda.empty_cache()

        try:
            nemo_ckpt_path = hf_hub_download(
                repo_id=settings.nemo_repo_id,
                filename=settings.nemo_filename,
                cache_dir=settings.model_cache_dir,
                token=settings.hf_token,
            )

            extracted_tokenizer_path = (
                settings.model_cache_dir / "indic_tokenizer.model"
            )
            extracted_vocab_path = settings.model_cache_dir / "vocab.txt"
            extracted_config_path = settings.model_cache_dir / "model_config.yaml"
            extracted_weights_path = settings.model_cache_dir / "model_weights.ckpt"

            # 1. Extract Files
            with tarfile.open(nemo_ckpt_path) as tar:

                def extract(member_name, dest):
                    m = next(
                        (m for m in tar.getmembers() if member_name in m.name), None
                    )
                    if m:
                        with open(dest, "wb") as f:
                            shutil.copyfileobj(tar.extractfile(m), f)
                        return True
                    return False

                extract(".model", extracted_tokenizer_path)
                if not extract("vocab.txt", extracted_vocab_path):
                    if not extracted_vocab_path.exists():
                        with open(extracted_vocab_path, "w") as f:
                            f.write("")
                extract(".yaml", extracted_config_path)
                extract(".ckpt", extracted_weights_path)

            # 2. Load as Plain YAML (Sanitized)
            with open(extracted_config_path, "r", encoding="utf-8") as f:
                full_conf_dict = yaml.safe_load(f)

            # 3. SMART SEARCH: Find the real model config
            # This avoids ambiguous nesting issues (model.model vs model etc.)
            model_cfg = self._find_model_config_root(full_conf_dict)
            if model_cfg is None:
                # Fallback: assume top level is model
                model_cfg = full_conf_dict

            # 4. PATCH THE CONFIG
            if "target" in model_cfg:
                del model_cfg["target"]

            # Remove training data paths
            for key in ["train_ds", "validation_ds", "test_ds"]:
                if key in model_cfg:
                    del model_cfg[key]

            # Inject Tokenizer
            if "tokenizer" not in model_cfg:
                model_cfg["tokenizer"] = {}
            model_cfg["tokenizer"]["dir"] = str(settings.model_cache_dir)
            model_cfg["tokenizer"]["type"] = "bpe"
            model_cfg["tokenizer"]["model_path"] = str(
                extracted_tokenizer_path.resolve()
            )
            model_cfg["tokenizer"]["vocab_path"] = str(extracted_vocab_path.resolve())

            # Map Decoder (Hybrid -> CTC)
            if "ctc_decoder" in model_cfg:
                model_cfg["decoder"] = model_cfg["ctc_decoder"]
            elif "aux_ctc" in model_cfg:
                model_cfg["decoder"] = model_cfg["aux_ctc"]

            # Inject num_classes
            if "num_classes" not in model_cfg.get("decoder", {}):
                try:
                    with open(extracted_vocab_path, "r", encoding="utf-8") as f:
                        lines = len(f.readlines())
                    if "decoder" not in model_cfg:
                        model_cfg["decoder"] = {}
                    model_cfg["decoder"]["num_classes"] = lines if lines > 0 else 1024
                    model_cfg["decoder"]["vocabulary"] = []
                except:
                    if "decoder" not in model_cfg:
                        model_cfg["decoder"] = {}
                    model_cfg["decoder"]["num_classes"] = 1024

            # FORCE INJECT PREPROCESSOR
            # Overwrite even if it exists to ensure it has valid params for this container
            model_cfg["preprocessor"] = {
                "_target_": "nemo.collections.asr.modules.AudioToMelSpectrogramPreprocessor",
                "sample_rate": 16000,
                "normalize": "per_feature",
                "window_size": 0.025,
                "window_stride": 0.01,
                "features": 80,
                "n_fft": 512,
                "dither": 0.00001,
                "pad_to": 16,
                "stft_conv": False,
            }

            # 5. Create OmegaConf (Unstructured)
            # We wrap ONLY the model config we found/patched
            final_conf = OmegaConf.create(model_cfg)
            OmegaConf.set_struct(final_conf, False)

            # 6. Instantiate
            print("[INFO] Instantiating EncDecCTCModelBPE manually...")
            model = EncDecCTCModelBPE(cfg=final_conf, trainer=None)

            # 7. Load Weights
            print(f"[INFO] Loading weights from {extracted_weights_path}...")
            if extracted_weights_path.exists():
                checkpoint = torch.load(
                    extracted_weights_path, map_location=self.device
                )
                state_dict = checkpoint.get("state_dict", checkpoint)
                model.load_state_dict(state_dict, strict=False)
            else:
                print("[ERROR] Weights file not found.")

            model.freeze()
            model.to(torch.device(self.device))
            self._nemo_model = model
            self._current_model_id = "ai4bharat/indicconformer"
            print("[SUCCESS] Loaded IndicConformer (Manual CTC Mode).")

        except Exception as e:
            raise RuntimeError(f"Failed to load NeMo model: {e}") from e

    def _generate_nemo_chunks(
        self, text: str, audio_path: Path
    ) -> list[dict[str, Any]]:
        duration = self._get_audio_duration(audio_path)
        words = text.split()
        if not words:
            return []
        total_chars = sum(len(w) for w in words) or 1
        max_chars_per_segment = 80
        chunks = []
        current_words = []
        current_chars = 0
        char_so_far = 0

        for w in words:
            wlen = len(w)
            if (current_chars + wlen + 1 > max_chars_per_segment) and current_words:
                seg_text = " ".join(current_words)
                chunks.append({"text": seg_text, "char_start": char_so_far})
                char_so_far += len(seg_text.replace(" ", ""))
                current_words = [w]
                current_chars = wlen
            else:
                current_words.append(w)
                current_chars += wlen + (1 if current_words else 0)

        if current_words:
            chunks.append({"text": " ".join(current_words), "char_start": char_so_far})

        formatted_chunks = []
        for ch in chunks:
            seg_text = ch["text"]
            start_t = duration * (ch["char_start"] / total_chars)
            seg_len = len(seg_text.replace(" ", ""))
            end_t = duration * ((ch["char_start"] + seg_len) / total_chars)
            if end_t - start_t < 1.5:
                end_t = min(start_t + 1.5, duration)
            formatted_chunks.append({"text": seg_text, "timestamp": (start_t, end_t)})
        return formatted_chunks

    def _convert_and_cache_whisper(self, model_id: str) -> str:
        sanitized_name = model_id.replace("/", "_")
        ct2_output_dir = settings.model_cache_dir / "converted_models" / sanitized_name
        raw_model_dir = settings.model_cache_dir / "raw_models" / sanitized_name
        if (ct2_output_dir / "model.bin").exists():
            return str(ct2_output_dir)

        print(f"[INFO] Converting {model_id} to CTranslate2...")
        snapshot_download(
            repo_id=model_id,
            local_dir=str(raw_model_dir),
            local_dir_use_symlinks=False,
            max_workers=8,
            token=settings.hf_token,
            ignore_patterns=["*.msgpack", "*.h5", "*.tflite", "*.ot"],
        )
        quant_type = "float16" if self.device == "cuda" else "int8"
        converter = ctranslate2.converters.TransformersConverter(
            str(raw_model_dir), load_as_float16=(self.device == "cuda")
        )
        converter.convert(str(ct2_output_dir), quantization=quant_type, force=True)
        for fname in ["tokenizer.json", "vocab.json", "preprocessor_config.json"]:
            src = raw_model_dir / fname
            if src.exists():
                shutil.copy(src, ct2_output_dir / fname)
        return str(ct2_output_dir)

    def _load_whisper_model(self, model_key: str) -> None:
        if self._nemo_model:
            del self._nemo_model
            self._nemo_model = None
            gc.collect()
            if self.device == "cuda":
                torch.cuda.empty_cache()
        if self._model and self._current_model_id == model_key:
            return

        print(f"[INFO] Requesting Whisper Model: {model_key}...")
        final_model_path = self._convert_and_cache_whisper(model_key)
        try:
            self._model = WhisperModel(
                final_model_path,
                device=self.device,
                compute_type=self.compute_type,
                download_root=str(settings.model_cache_dir),
            )
            self._batched_model = BatchedInferencePipeline(model=self._model)
            self._current_model_id = model_key
            print(f"[SUCCESS] Loaded {model_key}")
        except Exception as e:
            raise RuntimeError(f"Failed to load Faster-Whisper: {e}") from e

    def _format_timestamp(self, seconds: float) -> str:
        hours, remainder = divmod(seconds, 3600)
        minutes, secs = divmod(remainder, 60)
        millis = int(round((secs - int(secs)) * 1000))
        return f"{int(hours):02}:{int(minutes):02}:{int(secs):02},{millis:03}"

    def _write_srt(
        self, chunks: list[dict[str, Any]], path: Path, offset: float
    ) -> int:
        count = 0
        last_text = ""
        with open(path, "w", encoding="utf-8") as f:
            for chunk in chunks:
                text = chunk.get("text", "").strip()
                timestamp = chunk.get("timestamp")
                if not text or not timestamp:
                    continue
                try:
                    start, end = timestamp
                except ValueError:
                    continue
                if start is None:
                    continue
                if end is None:
                    end = start + 2.0
                start, end = float(start), float(end)
                if (end - start) < 0.2:
                    continue
                if text == last_text:
                    continue
                f.write(
                    f"{count + 1}\n{self._format_timestamp(start + offset)} --> "
                    f"{self._format_timestamp(end + offset)}\n{text}\n\n"
                )
                count += 1
                last_text = text
        return count

    def transcribe(
        self,
        raw_audio_path: str,
        language: str | None = None,
        subtitle_path: Path | None = None,
        output_path: Path | None = None,
        start_time: float = 0.0,
        end_time: float | None = None,
    ) -> list[dict[str, Any]] | None:
        audio_path = self._resolve_path(raw_audio_path)
        if not audio_path.exists():
            print(f"[ERROR] Input file not found: {audio_path}")
            return None
        lang = language or settings.language
        out_srt = output_path or audio_path.with_suffix(".srt")

        if start_time == 0.0 and end_time is None:
            if self._find_existing_subtitles(audio_path, out_srt, subtitle_path, lang):
                return None

        proc_path = audio_path
        is_sliced = False
        if start_time > 0 or end_time is not None:
            proc_path = self._slice_audio(audio_path, start_time, end_time)
            is_sliced = True

        chunks = []
        candidates = settings.whisper_model_map.get(lang, [settings.fallback_model_id])

        for model_to_use in candidates:
            try:
                is_nemo_target = "ai4bharat" in model_to_use and lang == "ta"
                if is_nemo_target:
                    self._load_nemo_model()
                    if self._nemo_model is None:
                        raise RuntimeError("NeMo load failed")
                    wav_path = self._prepare_16k_mono_wav(proc_path)
                    try:
                        # CTC Transcribe
                        texts = self._nemo_model.transcribe(
                            [str(wav_path)], batch_size=1, language_id="ta"
                        )
                        raw_text = ""
                        if texts:
                            first = texts[0]
                            if isinstance(first, list):
                                raw_text = (
                                    first[0]["text"]
                                    if (
                                        first
                                        and isinstance(first[0], dict)
                                        and "text" in first[0]
                                    )
                                    else (first[0] if first else "")
                                )
                            elif isinstance(first, dict) and "text" in first:
                                raw_text = first["text"]
                            else:
                                raw_text = str(first)

                        clean_text = self._clean_tamil_text(raw_text)
                        chunks = self._generate_nemo_chunks(clean_text, wav_path)
                        print("[SUCCESS] NeMo Transcription complete.")
                        break
                    finally:
                        if wav_path.exists():
                            wav_path.unlink()
                else:
                    self._load_whisper_model(model_to_use)
                    if self._batched_model is None:
                        raise RuntimeError("Whisper load failed")
                    prompt = "வணக்கம். Hello sir." if lang == "ta" else None
                    segments, info = self._batched_model.transcribe(
                        str(proc_path),
                        batch_size=settings.batch_size,
                        language=lang,
                        initial_prompt=prompt,
                        vad_filter=True,
                    )
                    chunks = [
                        {"text": s.text.strip(), "timestamp": (s.start, s.end)}
                        for s in segments
                    ]
                    print(
                        f"[SUCCESS] Whisper Transcription complete. Prob: {info.language_probability}"
                    )
                    chunks = self._split_long_chunks(
                        chunks, max_segment_s=settings.chunk_length_s
                    )
                    break
            except Exception as e:
                print(f"[WARN] Failed with model {model_to_use}: {e}. Trying next...")
                continue

        if is_sliced and proc_path.exists():
            try:
                proc_path.unlink()
            except Exception:
                pass

        if chunks:
            lines = self._write_srt(chunks, out_srt, offset=start_time)
            print(f"[SUCCESS] Saved {lines} subtitles to: {out_srt}")
            return chunks
        else:
            print("[WARN] No speech detected or all models failed.")
            return None


def main() -> None:
    if len(sys.argv) < 2:
        sys.exit(1)
    args = sys.argv
    start = float(args[2]) if len(args) > 2 else 0.0
    end = float(args[3]) if len(args) > 3 and args[3].lower() != "none" else None
    sub = Path(args[4]) if len(args) > 4 and args[4].lower() != "none" else None
    lang = args[5] if len(args) > 5 else "ta"
    AudioTranscriber().transcribe(args[1], lang, sub, None, start, end)


if __name__ == "__main__":
    main()
