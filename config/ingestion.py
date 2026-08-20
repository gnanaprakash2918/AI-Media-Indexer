from pathlib import Path
from pydantic import Field


class IngestionSettings:
    """Settings for all data extraction and chunking pipelines."""
    
    frame_interval: float = Field(default=0.5, description="Seconds between frames (0.5=2fps, 1.0=1fps)")
    language: str | None = "ta"
    
    whisper_model_map: dict[str, list[str]] = {
        "ta": [
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
    fallback_model_id: str = "Systran/faster-whisper-small"
    
    frame_sample_ratio: int = Field(
        default=1,
        description="Process every Nth extracted frame (1=all for max accuracy)",
    )
    
    # Face Detection Settings
    face_detection_threshold: float = Field(default=0.3)
    face_detection_resolution: int = Field(default=960)
    face_clustering_threshold: float = Field(default=0.5)
    face_min_bbox_size: int = Field(default=32)
    face_min_det_score: float = Field(default=0.5)
    face_nms_threshold: float = Field(default=0.3)
    
    # Voice Intelligence
    enable_voice_analysis: bool = True
    pyannote_model: str = "pyannote/speaker-diarization-3.1"
    voice_embedding_model: str = "pyannote/wespeaker-voxceleb-resnet34-LM"
    min_speakers: int | None = None
    max_speakers: int | None = None
    voice_clustering_threshold: float = Field(default=0.7)
    
    # HDBSCAN Tuning
    hdbscan_min_cluster_size: int = Field(default=2)
    hdbscan_min_samples: int = Field(default=2)
    hdbscan_cluster_selection_epsilon: float = Field(default=0.3)
    
    # Audio Processing
    audio_rms_silence_db: float = Field(default=-60.0)
    whisper_language_lock: bool = Field(default=True)
    
    # Face Track Builder
    face_track_iou_threshold: float = Field(default=0.3)
    face_track_cosine_threshold: float = Field(default=0.5)
    face_track_max_missing_frames: int = Field(default=5)
    face_audio_sync_tolerance: float = Field(default=0.3)
    
    # Frame Deduplication & Scene Detection
    frame_dedup_threshold: float = Field(default=0.98)
    scene_detect_threshold: float = Field(default=15.0)
    scene_detect_min_length: float = Field(default=1.0)
    
    # Audio events track (CLAP / AST)
    enable_audio_events: bool = Field(default=True)
    clap_window_seconds: float = Field(default=5.0)
    clap_stride_seconds: float = Field(default=2.5)
    clap_detection_threshold: float = Field(default=0.25)
    ast_detection_threshold: float = Field(default=0.15)
    clap_model_id: str = Field(default="laion/clap-htsat-unfused")
    ast_model_id: str = Field(default="mit/ast-finetuned-audioset-10-10-0.4593")

    auto_detect_language: bool = Field(default=True)
    
    # SAM3
    enable_sam3_tracking: bool = Field(default=True)
    sam_checkpoint: str = Field(default="sam3.pt")
    sam_config: str = Field(default="config.json")
    
    summary_scene_duration: int = Field(default=300)
    auto_summarize_on_ingest: bool = Field(default=False)
    
    # Biometrics
    biometric_threshold: float = Field(default=0.6)
    insightface_model: str = Field(default="buffalo_sc")
    
    music_dominance_ratio: float = Field(default=0.6)
    
    # Video Models
    enable_internvideo: bool = Field(default=True)
    enable_languagebind: bool = Field(default=False)
    
    # Feature Flags
    enable_face_recognition: bool = Field(default=True)
    enable_ocr: bool = Field(default=True)
    enable_object_detection: bool = Field(default=True)
    enable_audio_analysis: bool = Field(default=False)
    enable_speech_emotion: bool = Field(default=False)
    
    # OCR
    ocr_engine: str = Field(default="paddle")
    ocr_language: str = Field(default="multilingual")
    ocr_skip_unchanged_frames: bool = Field(default=True)
    ocr_batch_size: int = Field(default=8)
    ocr_keyframes_only: bool = Field(default=False)
    ocr_throttle_seconds: float = Field(default=2.0)
    
    perceptual_hash_threshold: int = Field(default=8)
    yolo_confidence_threshold: float = Field(default=0.3)
    dominant_color_clusters: int = Field(default=3)
    thumbnail_time_seconds: float = Field(default=5.0)
    dialogue_segment_limit: int = Field(default=50)
    segment_fallback_duration: float = Field(default=2.0)
    
    # Memory chunking
    enable_chunking: bool = Field(default=True)
    chunk_duration_seconds: int = Field(default=600)
    min_media_length_for_chunking: int = Field(default=1800)
    auto_chunk_by_hardware: bool = Field(default=True)
    
    ingestion_stage_version: str = Field(default="v1.0.0")
    
    high_performance_mode: bool = Field(default=True)
    max_concurrent_jobs: int = Field(default=1)
    lazy_unload: bool = Field(default=True)

    @property
    def arcface_model_path(self) -> Path:
        """Absolute path to ArcFace model."""
        # Requires self.model_cache_dir from Settings composition
        return self.model_cache_dir / "arcface" / "w600k_r50.onnx"
