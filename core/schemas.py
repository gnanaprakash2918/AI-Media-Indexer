"""Pydantic models and enums for describing media assets."""

from datetime import datetime
from enum import Enum

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
