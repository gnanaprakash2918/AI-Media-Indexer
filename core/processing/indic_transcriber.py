"""AI4Bharat IndicConformer Pipeline via NVIDIA NeMo."""

import gc
from pathlib import Path
from typing import Any

import torch

try:
    import nemo.collections.asr as nemo_asr # type: ignore
except ImportError:
    nemo_asr = None

from config import settings
from core.utils.logger import log


class IndicASRPipeline:
    """IndicConformer ASR for Indic languages using NeMo toolkit."""
    
    MODEL_MAP = {
        "ta": "ai4bharat/indicconformer_stt_ta_hybrid_rnnt",
        "hi": "ai4bharat/indicconformer_stt_hi_hybrid_rnnt",
        "te": "ai4bharat/indicconformer_stt_te_hybrid_rnnt",
        "ml": "ai4bharat/indicconformer_stt_ml_hybrid_rnnt",
        "kn": "ai4bharat/indicconformer_stt_kn_hybrid_rnnt",
        "bn": "ai4bharat/indicconformer_stt_bn_hybrid_rnnt",
        "gu": "ai4bharat/indicconformer_stt_gu_hybrid_rnnt",
        "mr": "ai4bharat/indicconformer_stt_mr_hybrid_rnnt",
        "or": "ai4bharat/indicconformer_stt_or_hybrid_rnnt",
        "pa": "ai4bharat/indicconformer_stt_pa_hybrid_rnnt",
    }
    
    def __init__(self, lang: str = "ta"):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = None
        self.lang = lang
        self._model_name = self.MODEL_MAP.get(lang, self.MODEL_MAP["ta"])

    def load_model(self) -> None:
        if self.model is not None:
            return
            
        if nemo_asr is None:
            raise ImportError("NeMo toolkit not installed. Run: pip install nemo_toolkit[asr]")

        log(f"[IndicASR] Loading {self._model_name}...")
        
        try:
            self.model = nemo_asr.models.EncDecRNNTBPEModel.from_pretrained(
                model_name=self._model_name
            )
            self.model.to(self.device)
            self.model.eval()
            log(f"[IndicASR] Loaded on {self.device}")
        except Exception as e:
            log(f"[IndicASR] Failed to load: {e}")
            raise

    def transcribe(
        self, 
        audio_path: Path,
        language: str | None = None,
        **kwargs
    ) -> list[dict[str, Any]]:
        """Transcribe audio file and return segments.
        
        Args:
            audio_path: Path to audio file (will be converted to 16kHz if needed).
            language: Override language for this call.
            
        Returns:
            List of dicts with text and timestamp info.
        """
        if language and language != self.lang:
            self.lang = language
            self._model_name = self.MODEL_MAP.get(language, self.MODEL_MAP["ta"])
            self.model = None
            
        self.load_model()
        
        audio_path = Path(audio_path)
        if not audio_path.exists():
            log(f"[IndicASR] File not found: {audio_path}")
            return []

        log(f"[IndicASR] Transcribing {audio_path.name}...")
        
        try:
            with torch.no_grad():
                if self.model:
                    transcriptions = self.model.transcribe(
                        paths2audio_files=[str(audio_path)]
                    )
                else:
                    return []
                
            text = transcriptions[0] if transcriptions else ""
            
            return [{
                "text": text,
                "start": 0.0,
                "end": None,
                "type": "indic_transcription",
                "language": self.lang,
            }]
            
        except Exception as e:
            log(f"[IndicASR] Transcription failed: {e}")
            return []
        finally:
            self._cleanup()

    def _cleanup(self) -> None:
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()
        
    def unload_model(self) -> None:
        if self.model is not None:
            del self.model
            self.model = None
            self._cleanup()
            log("[IndicASR] Model unloaded")
