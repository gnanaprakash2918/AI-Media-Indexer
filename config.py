"""Configuration settings for the ASR and LLM pipeline."""

import sys
from enum import Enum
from pathlib import Path
from typing import Literal

import torch
from pydantic import Field, SecretStr, computed_field
from pydantic_settings import BaseSettings, SettingsConfigDict


class LLMProvider(str, Enum):
    """Supported LLM providers."""

    GEMINI = "gemini"
    OLLAMA = "ollama"


class Settings(BaseSettings):
    """Application settings, hardware config, and external keys."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    @staticmethod
    def project_root(start: Path | None = None) -> Path:
        """Find the project root directory."""
        start = start or Path(__file__).resolve()
        for parent in start.parents:
            if (parent / ".git").exists() or (parent / "pyproject.toml").exists():
                return parent
        raise RuntimeError("Project root not found")

    @computed_field
    @property
    def cache_dir(self) -> Path:
        """Central location for all caches (__pycache__, models, temp files)."""
        path = self.project_root() / ".cache"
        path.mkdir(exist_ok=True)
        return path

    @computed_field
    @property
    def model_cache_dir(self) -> Path:
        """Directory for model weights."""
        path = self.project_root() / "models"
        path.mkdir(exist_ok=True)
        return path

    @computed_field
    @property
    def prompt_dir(self) -> Path:
        """Directory for prompt templates."""
        path = self.project_root() / "prompts"
        path.mkdir(exist_ok=True)
        return path

    @computed_field
    @property
    def log_dir(self) -> Path:
        """Path to project_root/logs."""
        path = self.project_root() / "logs"
        path.mkdir(exist_ok=True)
        return path

    #  Infrastructure (Qdrant)
    qdrant_host: str = Field(default="localhost", description="Qdrant host")
    qdrant_port: int = Field(default=6333, description="Qdrant HTTP port")
    qdrant_backend: str = Field(default="docker", description="'memory' or 'docker'")

    #  Agent & LLM
    agent_model: str = Field(default="llama3.1", description="Model for Agent CLI")
    llm_provider: LLMProvider = Field(default=LLMProvider.OLLAMA)
    ollama_base_url: str = Field(default="http://localhost:11434")
    ollama_vision_model: str = Field(default="llava:7b")

    gemini_api_key: SecretStr | None = Field(
        default=None, validation_alias="GOOGLE_API_KEY"
    )
    gemini_model: str = "gemini-1.5-flash"

    ollama_base_url: str = "http://localhost:11434"
    ollama_model: str = "moondream"  # Lightweight vision model (~2GB VRAM vs 5GB for llava)

    tmdb_api_key: str | None = None
    omdb_api_key: str | None = None
    hf_token: str | None = Field(default=None, validation_alias="HF_TOKEN")

    frame_interval: float = Field(default=1.0, description="Seconds between frames (0.5=2fps, 1.0=1fps)")
    batch_size: int = Field(default=24)
    device_override: Literal["cuda", "cpu", "mps"] | None = None

    language: str | None = "ta"
    whisper_model_map: dict[str, list[str]] = {
        "ta": [
            # "ai4bharat/indicconformer",
            "openai/whisper-large-v3-turbo",
            "openai/whisper-large-v2",
            "openai/whisper-medium",  # ~1.5GB, good balance
            "openai/whisper-small",   # ~500MB, faster and lighter
        ],
        "en": [
            "openai/whisper-large-v3-turbo",
            "openai/whisper-large-v2",
            "distil-whisper/distil-large-v3",
            "distil-whisper/distil-medium.en",  # ~500MB, optimized for English
            "openai/whisper-small",   # ~500MB, multilingual fallback
        ],
    }

    # Fallback for memory-constrained systems
    fallback_model_id: str = "openai/whisper-small"

    # Frame Processing Settings
    frame_sample_ratio: int = Field(
        default=1, 
        description="Process every Nth extracted frame (1=all for max accuracy)"
    )
    
    # Face Detection Settings
    face_detection_threshold: float = Field(
        default=0.4,
        description="Face detection confidence threshold (0.3-0.9, lower=more faces)"
    )
    face_detection_resolution: int = Field(
        default=960,
        description="Face detection input resolution (320=fast, 640=balanced, 960=accurate)"
    )
    
    # Face Clustering Settings
    face_clustering_threshold: float = Field(
        default=0.45,
        description="Face clustering cosine distance (lower=stricter, 0.45=55% similarity required - balanced for wild videos)"
    )
    face_min_bbox_size: int = Field(
        default=48,
        description="Minimum face bounding box size in pixels for clustering"
    )
    face_min_det_score: float = Field(
        default=0.65,
        description="Minimum face detection confidence for clustering (0.5-0.8)"
    )

    # Voice Intelligence
    enable_voice_analysis: bool = True
    pyannote_model: str = "pyannote/speaker-diarization-3.1"
    voice_embedding_model: str = "pyannote/wespeaker-voxceleb-resnet34-LM"
    min_speakers: int | None = None
    max_speakers: int | None = None
    voice_clustering_threshold: float = Field(
        default=0.45,
        description="Voice clustering cosine distance (lower=stricter, 0.45=55% similarity required)"
    )
    
    # Audio Processing
    audio_rms_silence_db: float = Field(
        default=-40.0,
        description="RMS threshold in dB below which audio is considered silent"
    )
    whisper_language_lock: bool = Field(
        default=True,
        description="Lock Whisper to detected language after first 30s"
    )
    
    # Face Track Builder
    face_track_iou_threshold: float = Field(default=0.3, description="Min IoU for face track continuity")
    face_track_cosine_threshold: float = Field(default=0.5, description="Min cosine sim for face track")
    face_track_max_missing_frames: int = Field(default=5, description="Frames before track finalization")
    
    # Scene Detection
    scene_detect_threshold: float = Field(default=27.0, description="PySceneDetect threshold (lower=more scenes)")
    scene_detect_min_length: float = Field(default=1.0, description="Min scene length in seconds")
    
    # AI Provider Strategy (runtime switchable)
    ai_provider_vision: str = Field(
        default="ollama",
        description="VLM provider for dense captioning (ollama/gemini)"
    )
    ai_provider_text: str = Field(
        default="ollama",
        description="LLM provider for query parsing (ollama/gemini)"
    )

    # Resource
    enable_resource_monitoring: bool = True
    max_cpu_percent: float = 90.0
    max_ram_percent: float = 85.0
    max_temp_celsius: float = 85.0  # Pause if CPU hits 85°C

    # Pause duration when overheated (seconds)
    cool_down_seconds: int = 30

    # Langfuse Configuration
    langfuse_backend: Literal["docker", "cloud", "disabled"] = Field(
        default="disabled",
        description="Langfuse backend selection",
    )

    # Cloud Langfuse
    langfuse_public_key: str | None = None
    langfuse_secret_key: str | None = None
    langfuse_host: str = "https://cloud.langfuse.com"

    # Local (Docker) Langfuse
    langfuse_docker_host: str = "http://localhost:3300"
    
    # --- Antigravity Feature Flags ---
    use_indic_asr: bool = Field(
        default=True, 
        description="Use AI4Bharat IndicConformer for Indic languages"
    )
    
    auto_detect_language: bool = Field(
        default=True,
        description="Auto-detect audio language before transcription"
    )
    
    enable_sam3_tracking: bool = Field(
        default=False,
        description="Enable SAM 3 Promptable Concept Segmentation"
    )
    
    manipulation_backend: Literal["disabled", "wan", "propainter", "auto"] = Field(
        default="disabled",
        description="Backend for video inpainting/manipulation"
    )

    # --- Biometrics Configuration ---
    arcface_model_path: Path = Field(
        default=Path("models/arcface/w600k_r50.onnx"),
        description="Path to ArcFace ONNX model for twin verification"
    )
    biometric_threshold: float = Field(
        default=0.6,
        description="Distance threshold for ArcFace identity verification"
    )

    # Advanced Overrides - SOTA Embeddings for 100% accuracy
    embedding_model_override: str = Field(
        default="BAAI/bge-m3",
        description="Text embedding model (BGE-M3 = 1024d, SOTA multilingual)"
    )
    
    siglip_model: str = Field(
        default="google/siglip-so400m-patch14-384",
        description="Visual embedding model for cross-modal search"
    )
    
    enable_visual_embeddings: bool = Field(
        default=True,
        description="Store SigLIP visual embeddings for cross-modal retrieval"
    )
    
    enable_hybrid_search: bool = Field(
        default=True,
        description="Use hybrid search (vector + keyword + identity)"
    )

    @computed_field
    @property
    def device(self) -> str:
        """Decide the device based on CPU or CUDA."""
        if self.device_override:
            return self.device_override
        if torch.cuda.is_available():
            return "cuda"
        if torch.backends.mps.is_available():
            return "mps"
        return "cpu"

    @computed_field
    @property
    def compute_type(self) -> str:
        """Determine the compute type (float16/int8) based on device."""
        if self.device == "cuda":
            return "float16"
        return "int8"

    @computed_field
    @property
    def device_index(self) -> list[int]:
        """List of available device indices."""
        if self.device == "cuda":
            return list(range(torch.cuda.device_count()))
        return []


settings = Settings()

sys.pycache_prefix = str(settings.cache_dir / "pycache")
