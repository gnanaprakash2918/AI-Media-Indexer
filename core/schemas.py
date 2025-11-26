"""Pydantic models and enums for describing media assets."""

from datetime import datetime
from enum import Enum
from typing import Any

from pydantic import BaseModel, Field, FilePath


class MediaType(str, Enum):
    """Supported media types for scanned files."""

    VIDEO = "video"
    AUDIO = "audio"
    IMAGE = "image"


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
    confidence: float = 1.0  # Placeholder for future use


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
