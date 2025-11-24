from pydantic import BaseModel, FilePath, Field
from datetime import datetime
from enum import Enum

class MediaType(str, Enum):
    VIDEO = "video"
    AUDIO = "audio"
    IMAGE = "image"

class MediaAsset(BaseModel):
    """The standard A2A data packet for a file that's found on scan."""

    file_path: FilePath
    media_type: MediaType
    file_size_bytes: int = Field(..., description="Size in bytes")
    last_modified: datetime