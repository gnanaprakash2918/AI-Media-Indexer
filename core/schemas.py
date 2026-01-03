"""Pydantic models and enums for describing media assets."""

from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, FilePath, field_validator


class MediaType(str, Enum):
    """Supported media types for scanned files."""

    MOVIE = "movie"
    TV = "tv"
    VIDEO = "video"
    AUDIO = "audio"
    IMAGE = "image"
    PERSONAL = "personal"
    UNKNOWN = "unknown"


# --- UNIVERSAL PERCEPTION SCHEMAS (For FAANG-level search) ---

class EntityDetail(BaseModel):
    """Details about ANY specific object, food, item, or tool in a frame."""
    name: str = Field(
        ..., 
        description="Specific name (e.g., 'Idly', 'iPhone 15 Pro', 'Katana', 'Tesla Model 3')"
    )
    category: str = Field(
        ..., 
        description="Category (e.g., 'Food', 'Electronics', 'Weapon', 'Vehicle')"
    )
    visual_details: str | dict | list = Field(
        default="", 
        description="Color, state, texture - accepts string or structured data from LLM"
    )
    
    def details_as_string(self) -> str:
        """Convert visual_details to string for search indexing."""
        if isinstance(self.visual_details, str):
            return self.visual_details
        if isinstance(self.visual_details, dict):
            return ", ".join(f"{k}: {v}" for k, v in self.visual_details.items())
        if isinstance(self.visual_details, list):
            parts = []
            for item in self.visual_details:
                if isinstance(item, dict):
                    parts.append(", ".join(f"{k}: {v}" for k, v in item.items()))
                else:
                    parts.append(str(item))
            return "; ".join(parts)
        return str(self.visual_details)


class SceneContext(BaseModel):
    """Contextual understanding of the scene."""
    location: str = Field(default="", description="Specific location")
    action_narrative: str = Field(default="", description="Precise action physics")
    cultural_context: str | None = Field(default=None, description="Inferred cultural setting")
    visible_text: list = Field(default_factory=list, description="Readable text/brands")
    
    def get_text_strings(self) -> list[str]:
        """Extract text strings from visible_text (handles dicts from LLM)."""
        result = []
        for item in self.visible_text:
            if isinstance(item, str):
                result.append(item)
            elif isinstance(item, dict):
                result.append(item.get("text", str(item)))
            else:
                result.append(str(item))
        return result


class FrameAnalysis(BaseModel):
    """Universal Structured Knowledge for ANY video frame. Accepts flexible LLM output."""
    main_subject: str | dict | list = Field(default="", description="Main person/object in focus")
    action: str | dict | list = Field(default="", description="Precise action occurring")
    action_physics: str | dict | list = Field(default="", description="Physical motion details")
    entities: list = Field(default_factory=list, description="Key objects")
    scene: SceneContext | dict = Field(default_factory=SceneContext)
    
    @field_validator("scene", mode="before")
    @classmethod
    def validate_scene(cls, v: Any) -> Any:
        """Handle Ollama returning a list for scene instead of a dict."""
        if isinstance(v, list) and len(v) > 0:
            return v[0]  # Extract first item if it's a list
        return v
    
    face_cluster_ids: list[int] = Field(default_factory=list, description="Face cluster IDs")
    
    def _to_str(self, val) -> str:
        """Convert any value to string for search indexing."""
        if isinstance(val, str):
            return val
        if isinstance(val, dict):
            return ", ".join(f"{k}: {v}" for k, v in val.items())
        if isinstance(val, list):
            return "; ".join(self._to_str(item) for item in val)
        return str(val) if val else ""
    
    def to_search_content(self) -> str:
        """Generate a rich semantic search string."""
        parts = []
        
        if self.main_subject:
            parts.append(self._to_str(self.main_subject))
        if self.action:
            parts.append(self._to_str(self.action))
        if self.action_physics:
            parts.append(self._to_str(self.action_physics))
        
        for e in self.entities:
            if isinstance(e, dict):
                name = e.get("name", "")
                details = e.get("visual_details", "")
                if name:
                    parts.append(f"{name} ({self._to_str(details)})" if details else name)
            elif hasattr(e, 'name'):
                entity_str = e.name
                if hasattr(e, 'details_as_string'):
                    entity_str += f" ({e.details_as_string()})"
                elif e.visual_details:
                    entity_str += f" ({self._to_str(e.visual_details)})"
                parts.append(entity_str)
        
        scene = self.scene if isinstance(self.scene, SceneContext) else SceneContext(**self.scene) if isinstance(self.scene, dict) else SceneContext()
        if scene.location:
            parts.append(scene.location)
        parts.extend(scene.get_text_strings() if hasattr(scene, 'get_text_strings') else [])
        if scene.cultural_context:
            parts.append(scene.cultural_context)
        
        return " ".join(filter(None, parts))


class MediaMetadata(BaseModel):
    """Extracted metadata from a media file."""
    year: int | None = None
    media_type: MediaType = MediaType.UNKNOWN

    cast: list[str] = Field(default_factory=list)
    director: str | None = None
    plot_summary: str | None = None

    is_processed: bool = False
    duration: float | None = None
    width: int | None = None
    height: int | None = None
    codec: str | None = None
    fps: float | None = None
    size_bytes: int | None = None
    created_at: str | None = None
    modified_at: str | None = None
    title: str | None = None
    artist: str | None = None
    album: str | None = None
    description: str | None = None


class UnresolvedFace(BaseModel):
    """Represents a face cluster that needs naming."""

    cluster_id: int
    face_encoding: list[float]
    sample_image_path: str
    occurrence_count: int
    suggested_name: str | None = None


class MediaAsset(BaseModel):
    """Metadata describing a discovered media file.

    Attributes:
      file_path: Absolute path to the media file.
      media_type: Detected type of the media file.
      file_size_bytes: File size in bytes.
      last_modified: Last modification timestamp of the file.
    """

    file_path: FilePath
    media_type: MediaType
    file_size_bytes: int = Field(..., description="Size in bytes")
    last_modified: datetime

class DetectedFace(BaseModel):
    """Represents a face detected in a media file."""

    bbox: tuple[int, int, int, int]
    confidence: float = 1.0
    landmarks: list[tuple[int, int]] | None = None
    embedding: list[float] | None = Field(
        default=None, exclude=True, description="128-dimensional face vector"
    )
    person_id: str | None = None

class TranscriptionResult(BaseModel):
    """Result of transcribing an audio file.

    Attributes:
        text: The full transcription text.
        segments: A list of segments with timing and text details.
        language: Detected language code (e.g., 'en').
        language_probability: Probability score for the detected language.
        duration: Duration of the audio in seconds.
    """

    text: str
    segments: list[dict[str, Any]]
    language: str
    language_probability: float
    duration: float


# Pydantic response models


class SearchResponse(BaseModel):
    """Response payload for media search.

    Attributes:
        visual_matches: Frame-based search hits, usually containing score,
            timestamp, file path, and a short description of the scene.
        dialogue_matches: Dialogue/text-based search hits, usually containing
            score, timestamp, file path, and a subtitle/transcript snippet.
    """

    visual_matches: list[dict[str, Any]] = Field(
        default_factory=list,
        description="Frame-based visual matches.",
    )
    dialogue_matches: list[dict[str, Any]] = Field(
        default_factory=list,
        description="Dialogue/text matches.",
    )


class IngestResponse(BaseModel):
    """Response payload for media ingestion.

    Attributes:
        file_path: Resolved absolute path of the ingested file.
        media_type_hint: Media type hint that was forwarded to the pipeline.
        message: Human-readable summary of the ingestion outcome.
    """

    file_path: str
    media_type_hint: str
    message: str


class ProcessingStatus(str, Enum):
    """Status of media processing workflow."""

    PENDING = "pending"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"

class SpeakerSegment(BaseModel):
    """A segment of audio attributed to a specific speaker."""

    start_time: float
    end_time: float
    speaker_label: str
    confidence: float = 0.0
    transcribed_text: str | None = None
    embedding: list[float] | None = Field(default=None, exclude=True)

    model_config = ConfigDict(from_attributes=True)


class DetectedVoice(BaseModel):
    """Metadata about a unique voice found in the media."""

    label: str
    embedding_avg: list[float]
    total_duration: float

class MediaFile(BaseModel):
    """Represents a media file in the system."""

    path: str
    filename: str
    media_type: MediaType
    content_hash: str
    metadata: MediaMetadata = Field(default_factory=MediaMetadata)

    # Analysis Data
    transcript: str | None = None
    summary: str | None = None
    visual_description: str | None = None
    detected_faces: list[DetectedFace] = Field(default_factory=list)
    speaker_segments: list[SpeakerSegment] = Field(default_factory=list)

    # System Data
    status: ProcessingStatus = ProcessingStatus.PENDING
    error_message: str | None = None
    turn_id: str | None = None

    model_config = ConfigDict(from_attributes=True)

    @property
    def file_path(self) -> Path:
        """Get the file path as a Path object."""
        return Path(self.path)


class MatchReason(str, Enum):
    IDENTITY_FACE = "identity_face"
    IDENTITY_VOICE = "identity_voice"
    SEMANTIC_VISUAL = "semantic_visual"
    SEMANTIC_AUDIO = "semantic_audio"
    VLM_VERIFIED = "vlm_verified"


class SearchResultDetail(BaseModel):
    video_id: str
    file_path: str
    start_time: float
    end_time: float
    score: float = Field(..., ge=0.0, le=1.0, description="Normalized confidence 0.0-1.0")
    match_reasons: list[MatchReason]
    explanation: str = Field(default="", description="VLM-generated explanation")
    thumbnail_url: str | None = None
    dense_context: str = ""
    matched_identities: list[str] = Field(default_factory=list)
