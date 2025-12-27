"""Pydantic models and enums for describing media assets."""

from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, FilePath


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
    """Details about ANY specific object, food, item, or tool in a frame.
    
    Forces AI to use SPECIFIC names instead of generic terms.
    Works for ANY category: Food, Vehicles, Electronics, Weapons, Clothing, etc.
    """
    name: str = Field(
        ..., 
        description="Specific name (e.g., 'Idly', 'iPhone 15 Pro', 'Katana', 'Tesla Model 3', 'Nike Air Jordan')"
    )
    category: str = Field(
        ..., 
        description="Category (e.g., 'Food', 'Electronics', 'Weapon', 'Vehicle', 'Footwear', 'Beverage', 'Furniture')"
    )
    visual_details: str = Field(
        default="", 
        description="Color, state, texture (e.g., 'Steaming hot', 'Cracked screen', 'Bloodied', 'Dented bumper')"
    )


class SceneContext(BaseModel):
    """Contextual understanding of the scene - works for ANY scene type."""
    location: str = Field(
        default="", 
        description="Specific location (e.g., 'Bowling Alley', 'Mars Colony', 'Operating Room', 'Tokyo Intersection')"
    )
    action_narrative: str = Field(
        default="", 
        description="Precise action physics (e.g., 'Pin wobbling before falling', 'Surgeon making incision', 'Car drifting')"
    )
    cultural_context: str | None = Field(
        default=None, 
        description="Inferred cultural setting (e.g., 'South Indian Breakfast', 'Japanese Tea Ceremony', 'American Football')"
    )
    visible_text: list[str] = Field(
        default_factory=list, 
        description="All readable text/brands (e.g., 'LensKart', 'Tesla', 'Brunswick', 'Nike', 'Stop')"
    )


class FrameAnalysis(BaseModel):
    """Universal Structured Knowledge for ANY video frame.
    
    This schema forces the AI to output SPECIFIC names instead of generics.
    Examples: 'Idly' not 'food', 'Katana' not 'weapon', 'Tesla Model 3' not 'car'
    """
    main_subject: str = Field(
        default="", 
        description="Main person or object in focus (e.g., 'Samurai warrior', 'Chef', 'Racing driver')"
    )
    action: str = Field(
        default="", 
        description="Precise action (e.g., 'eating idly', 'slashing with katana', 'drifting around corner')"
    )
    entities: list[EntityDetail] = Field(
        default_factory=list, 
        description="Key objects, foods, tools, vehicles, weapons, or brands"
    )
    scene: SceneContext = Field(default_factory=SceneContext)
    
    # Identity linking (filled by pipeline)
    face_cluster_ids: list[int] = Field(
        default_factory=list, 
        description="Face cluster IDs in this frame"
    )
    
    def to_search_content(self) -> str:
        """Generate a rich semantic search string with specific terms."""
        parts = []
        
        # Main subject and action
        if self.main_subject:
            parts.append(self.main_subject)
        if self.action:
            parts.append(self.action)
        
        # Entities with details (e.g., "Idly (Steaming hot)")
        for e in self.entities:
            entity_str = e.name
            if e.visual_details:
                entity_str += f" ({e.visual_details})"
            parts.append(entity_str)
        
        # Scene location
        if self.scene.location:
            parts.append(self.scene.location)
        
        # OCR text / brands
        parts.extend(self.scene.visible_text)
        
        # Cultural context
        if self.scene.cultural_context:
            parts.append(self.scene.cultural_context)
        
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
