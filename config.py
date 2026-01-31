"""Configuration settings for the ASR and LLM pipeline."""

# Windows compatibility patches (must be imported before third-party libs)
import core.utils.platform_compat  # noqa: F401, E402

import logging
import sys
from enum import Enum
from pathlib import Path
from typing import Literal

import torch
from pydantic import Field, SecretStr, computed_field, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class HardwareProfile(str, Enum):
    """Hardware profile for throughput tuning.

    NOTE: Profiles affect ONLY batch sizes and parallelism.
    Model quality and accuracy are IDENTICAL across all profiles.
    """

    LAPTOP = "laptop"  # < 8GB VRAM
    WORKSTATION = "workstation"  # 8-20GB VRAM
    SERVER = "server"  # > 20GB VRAM
    CPU_ONLY = "cpu_only"  # No GPU


def get_hardware_profile() -> dict:
    """Detect hardware and return optimal settings.

    Returns throughput settings based on available VRAM.
    NOTE: Model quality is NEVER reduced - only batch sizes change.
    """
    profile = {
        "name": HardwareProfile.CPU_ONLY,
        "batch_size": 4,
        "worker_count": 1,
        "embedding_batch_size": 4,
        "device": "cpu",
    }

    if not torch.cuda.is_available():
        logging.info(
            "No GPU detected. Using CPU profile (full accuracy, slower)."
        )
        return profile

    profile["device"] = "cuda"

    try:
        # Get VRAM in GB
        vram_bytes = torch.cuda.get_device_properties(0).total_memory
        vram_gb = vram_bytes / (1024**3)

        if vram_gb >= 20.0:  # Server (e.g. A100, 3090/4090 24GB)
            profile["name"] = HardwareProfile.SERVER
            profile["batch_size"] = 16
            profile["worker_count"] = 4
            profile["embedding_batch_size"] = 32
            logging.info(f"Detected SERVER profile ({vram_gb:.1f}GB VRAM)")
        elif vram_gb >= 8.0:  # Workstation (e.g. 3070/4070, 8-16GB)
            profile["name"] = HardwareProfile.WORKSTATION
            profile["batch_size"] = 8
            profile["worker_count"] = 2
            profile["embedding_batch_size"] = 16
            logging.info(f"Detected WORKSTATION profile ({vram_gb:.1f}GB VRAM)")
        else:  # Laptop / Low-end (< 8GB)
            profile["name"] = HardwareProfile.LAPTOP
            profile["batch_size"] = 4
            profile["worker_count"] = 1
            profile["embedding_batch_size"] = 8
            logging.info(f"Detected LAPTOP profile ({vram_gb:.1f}GB VRAM)")

    except Exception as e:
        logging.warning(f"Failed to detect detailed hardware specs: {e}")

    return profile


_HW_PROFILE = get_hardware_profile()


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
            if (parent / ".git").exists() or (
                parent / "pyproject.toml"
            ).exists():
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

    # --- Performance ---
    batch_size: int = Field(
        default=_HW_PROFILE["batch_size"],
        description="Batch size for inference",
    )

    embedding_batch_size: int = Field(
        default=_HW_PROFILE["embedding_batch_size"],
        description="Batch size for embedding generation",
    )

    worker_count: int = Field(
        default=_HW_PROFILE["worker_count"],
        description="Number of parallel ingestion workers",
    )

    # device field removed to avoid conflict with computed_property 'device'
    # Use device_override to manually set it.

    #  Infrastructure (Qdrant)
    qdrant_host: str = Field(default="localhost", description="Qdrant host")
    qdrant_port: int = Field(default=6333, description="Qdrant HTTP port")
    qdrant_backend: str = Field(
        default="docker", description="'memory' or 'docker'"
    )

    #  Agent & LLM
    agent_model: str = Field(
        default="llama3.1", description="Model for Agent CLI"
    )
    llm_provider: LLMProvider = Field(default=LLMProvider.OLLAMA)
    ollama_base_url: str = Field(
        default="http://localhost:11434",
        validation_alias="OLLAMA_BASE_URL",
        description="Base URL for Ollama API",
    )
    ollama_vision_model: str = Field(
        default="llava:7b",
        validation_alias="OLLAMA_VISION_MODEL",
        description="Vision model for image analysis (e.g., llava:7b, moondream, internlm2:7b)",
    )
    ollama_text_model: str = Field(
        default="llama3.1",
        validation_alias="OLLAMA_TEXT_MODEL",
        description="Text model for structured output (e.g., llama3.1, mistral)",
    )

    gemini_api_key: SecretStr | None = Field(
        default=None, validation_alias="GOOGLE_API_KEY"
    )
    gemini_model: str = "gemini-1.5-flash"

    tmdb_api_key: str | None = None
    omdb_api_key: str | None = None
    brave_api_key: str | None = Field(
        default=None, description="Brave Search API key for external enrichment"
    )
    enable_external_search: bool = Field(
        default=False,
        description="Enable external web search for unknown entities",
    )
    hf_token: str | None = Field(default=None, validation_alias="HF_TOKEN")

    frame_interval: float = Field(
        default=0.5, description="Seconds between frames (0.5=2fps, 1.0=1fps)"
    )
    # NOTE: batch_size is defined in Performance section (line 142) using hardware profile
    device_override: Literal["cuda", "cpu", "mps"] | None = None

    language: str | None = "ta"
    # Use pre-converted faster-whisper CTranslate2 models (no conversion needed)
    whisper_model_map: dict[str, list[str]] = {
        "ta": [
            # Pre-converted faster-whisper models (deepdml has the turbo version)
            "deepdml/faster-whisper-large-v3-turbo-ct2",
            "Systran/faster-whisper-large-v3",
            "Systran/faster-whisper-medium",
            "Systran/faster-whisper-small",
        ],
        "en": [
            "deepdml/faster-whisper-large-v3-turbo-ct2",
            "Systran/faster-whisper-large-v3",
            "Systran/faster-distil-whisper-large-v3",
            "Systran/faster-distil-whisper-medium.en",
            "Systran/faster-whisper-small",
        ],
    }

    # Fallback for memory-constrained systems (pre-converted)
    fallback_model_id: str = "Systran/faster-whisper-small"

    # Frame Processing Settings
    frame_sample_ratio: int = Field(
        default=1,
        description="Process every Nth extracted frame (1=all for max accuracy)",
    )

    # Face Detection Settings (tuned for side-facing and partial faces)
    face_detection_threshold: float = Field(
        default=0.3,
        description="Face detection confidence threshold (0.3=more faces including side-facing, 0.5=balanced)",
    )
    face_detection_resolution: int = Field(
        default=960,
        description="Face detection input resolution (320=fast, 640=balanced, 960=accurate)",
    )

    # Face Clustering Settings (tuned for wild videos with angle/lighting variations)
    face_clustering_threshold: float = Field(
        default=0.5,
        description="Face clustering cosine distance (0.5=50% similarity for same person across angles)",
    )
    face_min_bbox_size: int = Field(
        default=32,
        description="Minimum face bounding box size in pixels (32=include smaller/distant faces)",
    )
    face_min_det_score: float = Field(
        default=0.5,
        description="Minimum face detection confidence for clustering (0.5=include side-facing faces)",
    )

    # Voice Intelligence
    enable_voice_analysis: bool = True
    pyannote_model: str = "pyannote/speaker-diarization-3.1"
    voice_embedding_model: str = "pyannote/wespeaker-voxceleb-resnet34-LM"
    min_speakers: int | None = None
    max_speakers: int | None = None
    voice_clustering_threshold: float = Field(
        default=0.3,
        description="Voice clustering cosine distance (lower=stricter, 0.3=70% similarity required for tighter clusters)",
    )

    # HDBSCAN Tuning (used for clustering algorithms)
    hdbscan_min_cluster_size: int = Field(
        default=2,
        description="Minimum cluster size for HDBSCAN (2=pair of samples)",
    )
    hdbscan_min_samples: int = Field(
        default=2,
        description="Min samples for core point (2=reduces noise/singlets)",
    )
    hdbscan_cluster_selection_epsilon: float = Field(
        default=0.3,
        description="Cluster selection epsilon for HDBSCAN (0.3=strict, 0.45=balanced, 0.6=aggressive)",
    )

    # Audio Processing
    audio_rms_silence_db: float = Field(
        default=-60.0,
        description="RMS threshold in dB below which audio is considered silent",
    )
    whisper_language_lock: bool = Field(
        default=True,
        description="Lock Whisper to detected language after first 30s",
    )

    # Face Track Builder
    face_track_iou_threshold: float = Field(
        default=0.3, description="Min IoU for face track continuity"
    )
    face_track_cosine_threshold: float = Field(
        default=0.5, description="Min cosine sim for face track"
    )
    face_track_max_missing_frames: int = Field(
        default=5, description="Frames before track finalization"
    )

    # Scene Detection
    scene_detect_threshold: float = Field(
        default=15.0,
        description="PySceneDetect threshold (lower=more scenes, 15.0=sensitive for music videos)",
    )
    scene_detect_min_length: float = Field(
        default=1.0, description="Min scene length in seconds"
    )

    # AI Provider Strategy (runtime switchable)
    ai_provider_vision: str = Field(
        default="ollama",
        description="VLM provider for dense captioning (ollama/gemini)",
    )
    ai_provider_text: str = Field(
        default="ollama",
        description="LLM provider for query parsing (ollama/gemini)",
    )

    # Resource
    enable_resource_monitoring: bool = True
    max_cpu_percent: float = 90.0
    max_ram_percent: float = 95.0
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

    # Redis/Celery Configuration
    redis_host: str = "localhost"
    redis_port: int = 6379
    redis_auth: str = "redispass"
    enable_distributed_ingestion: bool = False

    # Observability (Loki)
    enable_loki: bool = Field(
        default=False, description="Enable log shipping to Grafana Loki"
    )
    loki_url: str = Field(
        default="http://localhost:3100/loki/api/v1/push",
        description="Loki push API URL",
    )

    # --- Antigravity Feature Flags ---
    use_indic_asr: bool = Field(
        default=True,
        description="Use AI4Bharat IndicConformer for Indic languages",
    )

    use_native_nemo: bool = Field(
        default=True,
        description="Attempt to use Native NeMo if installed (Preferred over Docker)",
    )

    auto_detect_language: bool = Field(
        default=True,
        description="Auto-detect audio language before transcription",
    )

    ai4bharat_url: str = Field(
        default="http://localhost:8001",
        description="URL for local AI4Bharat IndicConformer Docker service",
    )

    enable_sam3_tracking: bool = Field(
        default=True,
        description="Enable SAM 3 Promptable Concept Segmentation for frame-exact object tracking",
    )

    manipulation_backend: Literal["disabled", "wan", "propainter", "auto"] = (
        Field(
            default="disabled",
            description="Backend for video inpainting/manipulation",
        )
    )

    # Hierarchical Summarization
    summary_scene_duration: int = Field(
        default=300,
        description="Duration in seconds for L2 scene chunks (300=5 minutes)",
    )
    auto_summarize_on_ingest: bool = Field(
        default=False,
        description="Auto-generate hierarchical summaries after ingestion",
    )

    # --- Biometrics Configuration ---
    arcface_model_path: Path = Field(
        default=Path("models/arcface/w600k_r50.onnx"),
        description="Path to ArcFace ONNX model for twin verification",
    )
    biometric_threshold: float = Field(
        default=0.6,
        description="Distance threshold for ArcFace identity verification",
    )

    # Advanced Overrides - SOTA Embeddings for 100% accuracy
    embedding_model_override: str = Field(
        default="", # Empty = Auto-detect based on VRAM (SOTA preferred)
        description="Text embedding model (NV-Embed-v2 = 4096d, SOTA)",
    )
    text_embedding_dim: int = Field(
        default=4096, # Updated for NV-Embed-v2
        description="Dimension of text embeddings (must match model)",
    )
    visual_embedding_dim: int = Field(
        default=1152, # SigLIP SO400M is 1152 dim
        description="Dimension of visual embeddings ",
    )

    siglip_model: str = Field(
        default="google/siglip-so400m-patch14-384",
        description="Visual embedding model for cross-modal search",
    )

    enable_visual_embeddings: bool = Field(
        default=True,
        description="Store SigLIP visual embeddings for cross-modal retrieval",
    )

    # --- Frame VLM (per-frame dense captioning) ---
    enable_frame_vlm: bool = Field(
        default=True,
        description="Enable per-frame VLM captioning (fine-grained: faces, text, objects)",
    )

    # --- Video Understanding (InternVideo, LanguageBind) ---
    enable_video_embeddings: bool = Field(
        default=True,
        description="Compute InternVideo/LanguageBind embeddings for action/motion search",
    )
    
    # --- Hybrid VLM (best of both) ---
    enable_hybrid_vlm: bool = Field(
        default=True,
        description="Combine frame VLM + video VLM for complete understanding (recommended)",
    )
    
    video_embedding_dim: int = Field(
        default=1024,
        description="Dimension of video embeddings (InternVideo/LanguageBind projected)",
    )
    
    visual_features_dim: int = Field(
        default=1152,
        description="Dimension of visual features (SigLIP)",
    )

    enable_hybrid_search: bool = Field(
        default=True,
        description="Use hybrid search (vector + keyword + identity)",
    )

    enable_vlm_reranking: bool = Field(
        default=True,  # ENABLED for max quality
        description="Enable VLM-based reranking for higher accuracy (uses more VRAM and time)",
    )

    # --- Search Configuration (all previously hardcoded thresholds) ---
    # Query Expansion
    search_enable_expansion: bool = Field(
        default=True,
        description="Enable LLM query expansion (disable for non-English)",
    )
    search_expansion_fallback: bool = Field(
        default=True,
        description="Retry with original query if expansion yields few results",
    )
    search_expansion_min_results: int = Field(
        default=3,
        description="Min results before triggering fallback to original query",
    )

    # Hybrid Search Weights
    search_vector_weight: float = Field(
        default=0.5, # Balanced for better identity matching
        description="Weight for vector similarity in hybrid search (0.0-1.0)",
    )
    search_keyword_weight: float = Field(
        default=0.5,
        description="Weight for keyword/BM25 in hybrid search (0.0-1.0)",
    )

    # Retrieval Limits
    search_default_limit: int = Field(
        default=50, # Higher recall
        description="Default number of search results",
    )
    search_rerank_multiplier: int = Field(
        default=5, # Rerank top 250 for best precision
        description="Multiply limit by this for reranking pool (get 3x candidates)",
    )

    # Reranking Thresholds
    search_min_score_threshold: float = Field(
        default=0.2, # Lower threshold to let RRF/Reranker decide
        description="Minimum score to include in results (0.0-1.0)",
    )
    search_vlm_confidence_threshold: int = Field(
        default=60,
        description="VLM confidence threshold for verification (0-100)",
    )

    # HITL Feedback
    search_hitl_positive_boost: float = Field(
        default=1.5,
        description="Score multiplier for positive feedback (>1.0 = boost)",
    )
    search_hitl_negative_penalty: float = Field(
        default=0.5,
        description="Score multiplier for negative feedback (<1.0 = penalty)",
    )
    search_hitl_max_boost: float = Field(
        default=3.0,
        description="Maximum cumulative HITL boost",
    )
    search_hitl_min_penalty: float = Field(
        default=0.2,
        description="Minimum cumulative HITL penalty",
    )
    search_hitl_similarity_threshold: float = Field(
        default=0.7,
        description="Query similarity threshold for HITL feedback matching",
    )

    # Query Decomposition
    search_decomposition_confidence: float = Field(
        default=0.6,
        description="Min confidence for query decomposition constraints",
    )

    # RRF Fusion
    search_rrf_k: int = Field(
        default=60,
        description="RRF constant k (higher = more weight to lower ranks)",
    )

    # Temporal Context
    search_temporal_tolerance: float = Field(
        default=10.0,
        description="Seconds tolerance for temporal matching in feedback",
    )

    # Deep Research Cinematography Concepts (Configurable)
    # Allows users to inject their own film theory concepts without code changes
    cinematography_shot_types: list[str] = Field(
        default=[
            "close-up shot",
            "medium shot",
            "wide shot",
            "extreme close-up",
            "establishing shot",
            "over-the-shoulder shot",
            "point-of-view shot",
            "high angle shot",
            "low angle shot",
            "dutch angle shot",
            "aerial shot",
            "tracking shot",
        ],
        description="List of shot types for Zero-Shot classification",
    )

    cinematography_moods: list[str] = Field(
        default=[
            "happy",
            "sad",
            "tense",
            "romantic",
            "action-packed",
            "mysterious",
            "peaceful",
            "dramatic",
            "comedic",
            "horror",
            "melancholic",
            "euphoric",
        ],
        description="List of cinematic moods for Zero-Shot classification",
    )

    # Memory Management - STRATEGY: SOTA Always, Throttle Resources
    high_performance_mode: bool = Field(
        default=True,
        description="Enable parallel processing. If False, sequential processing with aggressive cleanup.",
    )

    max_concurrent_jobs: int = Field(
        default=1,
        description="Max parallel ingestion jobs. Auto-set by SystemProfile if not overridden.",
    )

    lazy_unload: bool = Field(
        default=True, description="Unload models after use to free VRAM."
    )

    # --- Memory Chunking (OOM Prevention) ---
    enable_chunking: bool = Field(
        default=True,
        description="Auto-chunk long videos/audio to prevent OOM. Always ON by default.",
    )
    chunk_duration_seconds: int = Field(
        default=600,
        description="Duration of each chunk in seconds (600 = 10 minutes)",
    )
    min_media_length_for_chunking: int = Field(
        default=1800,
        description="Only chunk if media > this length in seconds (1800 = 30 min). Set to 0 for always.",
    )
    auto_chunk_by_hardware: bool = Field(
        default=True,
        description="Auto-adjust chunking based on VRAM. Low VRAM = smaller chunks.",
    )

    # --- Search Feature Flags (Master Switches) ---
    enable_deep_research: bool = Field(
        default=True, description="Master switch for deep research (cinematography, aesthetics)."
    )
    enable_face_recognition: bool = Field(
        default=True, description="Master switch for face detection and clustering."
    )
    enable_ocr: bool = Field(
        default=True, description="Master switch for text extraction (OCR)."
    )
    enable_content_moderation: bool = Field(
        default=False,
        description="Master switch for NSFW/Safety checks (default OFF for speed).",
    )
    enable_time_extraction: bool = Field(
        default=False, description="Master switch for clock/time extraction (default OFF)."
    )
    enable_object_detection: bool = Field(
        default=True, description="Master switch for YOLO object detection."
    )

    # --- Audio Analysis (BPM/Beat Detection) ---
    enable_audio_analysis: bool = Field(
        default=False,
        description="Enable tempo/beat detection. OFF by default (only useful for music videos).",
    )

    # --- OCR Optimization ---
    ocr_skip_unchanged_frames: bool = Field(
        default=True,
        description="Skip OCR if frame is visually similar to previous (perceptual hash).",
    )
    ocr_batch_size: int = Field(
        default=8,
        description="Batch size for OCR processing (if using GPU-capable OCR).",
    )
    ocr_keyframes_only: bool = Field(
        default=False,
        description="Run OCR only on scene keyframes (faster but may miss some text).",
    )

    # --- Unified Search Configuration ---
    search_use_reasoning: bool = Field(
        default=False,
        description="Enable LLM query decomposition/reasoning. OFF by default for speed.",
    )
    search_use_reranking: bool = Field(
        default=False,
        description="Enable LLM re-ranking for higher accuracy. OFF by default for speed.",
    )
    search_auto_mode: bool = Field(
        default=True,
        description="Auto-select search mode based on query complexity.",
    )

    # --- Visual Encoder ---
    visual_encoder_type: str = Field(
        default="siglip",
        description="Primary visual encoder: 'siglip' (SOTA) or 'clip' (fallback).",
    )
    visual_encoder_fallback: bool = Field(
        default=True,
        description="Auto-fallback from SigLIP to CLIP on OOM.",
    )

    # --- Deep Research Optimization ---
    deep_research_per_scene: bool = Field(
        default=True,
        description="Run Deep Research (shot type, mood) on scene keyframes only, not every frame.",
    )

    @model_validator(mode="after")
    def adjust_dimensions(self) -> "Settings":
        """Auto-adjust embedding dimensions based on model name."""
        model = self.embedding_model_override.lower()

        if "nv-embed-v2" in model:
            # NV-Embed-v2 is 4096 dim
            if self.text_embedding_dim != 4096:
                logging.info(
                    "Auto-adjusting text_embedding_dim to 4096 for NV-Embed-v2"
                )
                self.text_embedding_dim = 4096
                # visual_embedding_dim is independent (SigLIP)

        elif "bge-m3" in model:
            if self.text_embedding_dim != 1024:
                self.text_embedding_dim = 1024
        elif "mxbai" in model:
            if self.text_embedding_dim != 1024:
                self.text_embedding_dim = 1024

        return self

    @computed_field
    @property
    def effective_embedding_model(self) -> str:
        """ALWAYS use SOTA embedding model. Never downgrade for quality."""
        return self.embedding_model_override  # NV-Embed-v2 (4096d) always

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
            return "float16" # SOTA precision
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
