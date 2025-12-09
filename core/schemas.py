"""Pydantic models and enums for describing media assets."""

from datetime import datetime
from enum import Enum
from typing import Any

from pydantic import BaseModel, Field, FilePath


class MediaType(str, Enum):
    """Supported media types for scanned files."""

    MOVIE = "movie"
    TV = "tv"
    VIDEO = "video"
    AUDIO = "audio"
    IMAGE = "image"
    PERSONAL = "personal"
    UNKNOWN = "unknown"


class MediaMetadata(BaseModel):
    """Describes metadata information for a media item such as a movie, audio, or image.

    Attributes:
        title: The title or name of the media item.
        year: The release or creation year of the media item, if known.
        media_type: The classification of the media (video, audio, image, etc.).
        cast: A list of names involved in the media (e.g., actors or participants).
        director: Name of the director or creator, if applicable.
        plot_summary: Brief summary or description of the media's content.
        is_processed: Flag indicating if the media metadata has been processed.
    """

    title: str
    year: int | None = None
    media_type: MediaType = MediaType.UNKNOWN

    cast: list[str] = Field(default_factory=list)
    director: str | None = None
    plot_summary: str | None = None

    is_processed: bool = False


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
    """Represents a single detected face in an image.

    Attributes:
        box:
            The bounding box as a 4-tuple (top, right, bottom, left) in pixel
            coordinates relative to the original image.
        encoding:
            The 128-d face embedding as a 1D numpy array of shape (128,).
    """

    box: tuple[int, int, int, int] = Field(
        ..., description="(top, right, bottom, left)"
    )
    encoding: list[float] = Field(..., description="128-dimensional face vector")
    confidence: float = 1.0


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
