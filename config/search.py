from pydantic import Field


class SearchSettings:
    """Settings for multimodal retrieval and ranking."""
    
    query_expansion_enabled: bool = Field(default=True)
    query_expansion_confidence: float = Field(default=0.7)
    
    rrf_constant: int = Field(default=60)
    timestamp_bucket_seconds: float = Field(default=2.0)
    rerank_vector_weight: float = Field(default=0.3)
    rerank_llm_weight: float = Field(default=0.7)
    
    # Base modality weights (sum should be ~1.0)
    modality_weight_scenes: float = Field(default=0.20)
    modality_weight_frames: float = Field(default=0.18)
    modality_weight_scenelets: float = Field(default=0.15)
    modality_weight_voice: float = Field(default=0.15)
    modality_weight_dialogue: float = Field(default=0.17)
    modality_weight_audio: float = Field(default=0.15)
    modality_weight_video_metadata: float = Field(default=0.05)
    
    voice_identity_boost: float = Field(default=1.5)
    scenelet_dedup_overlap: float = Field(default=0.8)
    
    # Search result display configuration
    search_default_duration: float = Field(default=5.0)
    search_padding_before: float = Field(default=3.0)
    search_padding_after: float = Field(default=3.0)
    
    # Scenelet Optimization
    scenelet_window_seconds: float = Field(default=10.0)
    scenelet_stride_seconds: float = Field(default=5.0)
    
    enable_hybrid_search: bool = Field(default=True)
    enable_vlm_reranking: bool = Field(default=True)
    
    # Query Expansion defaults
    search_enable_expansion: bool = Field(default=True)
    search_expansion_fallback: bool = Field(default=True)
    search_expansion_min_results: int = Field(default=3)
    
    # Hybrid Search Weights
    search_vector_weight: float = Field(default=0.5)
    search_keyword_weight: float = Field(default=0.5)
    
    # Retrieval Limits
    search_default_limit: int = Field(default=50)
    search_rerank_multiplier: int = Field(default=5)
    search_min_score_threshold: float = Field(default=0.2)
    search_vlm_confidence_threshold: int = Field(default=60)
    
    # HITL Feedback
    search_hitl_positive_boost: float = Field(default=1.5)
    search_hitl_negative_penalty: float = Field(default=0.5)
    search_hitl_max_boost: float = Field(default=3.0)
    search_hitl_min_penalty: float = Field(default=0.2)
    search_hitl_similarity_threshold: float = Field(default=0.7)
    
    # Query Decomposition
    search_decomposition_confidence: float = Field(default=0.6)
    search_rrf_k: int = Field(default=60)
    search_temporal_tolerance: float = Field(default=10.0)
    
    # Deep Research Cinematography Concepts
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
        ]
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
        ]
    )
    
    # Deduplication
    deduplication_window_seconds: float = Field(default=5.0)
    context_expansion_seconds: float = Field(default=3.5)
    
    # Model Mapping (for search result attribution)
    modality_model_map: dict[str, list[str]] = Field(
        default={
            "scenes": ["BGE-M3", "CLIP/SigLIP", "InternVideo", "LanguageBind"],
            "frames": ["BGE-M3", "CLIP/SigLIP"],
            "scenelets": ["BGE-M3"],
            "voice": ["BGE-M3", "Whisper"],
            "audio_events": ["BGE-M3", "CLAP"],
        }
    )
    
    # Unified Search Configuration
    search_use_reasoning: bool = Field(default=False)
    search_use_reranking: bool = Field(default=False)
    search_auto_mode: bool = Field(default=True)
    
    # Deep Research Optimization
    deep_research_per_scene: bool = Field(default=True)
