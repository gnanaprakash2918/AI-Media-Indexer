"""Collection names and dimension constants for the vector database."""

from config import settings
from core.utils.hardware import select_embedding_model

# Auto-select embedding model based on available VRAM
if settings.embedding_model_override:
    SELECTED_MODEL = settings.embedding_model_override
else:
    SELECTED_MODEL, _ = select_embedding_model()

# Collection names
MEDIA_SEGMENTS_COLLECTION = "media_segments"
MEDIA_COLLECTION = "media_frames"
FRAMES_COLLECTION = "media_frames"
FACES_COLLECTION = "faces"
VOICE_COLLECTION = "voice_segments"
SCENES_COLLECTION = "scenes"
SCENELETS_COLLECTION = "media_scenelets"
SUMMARIES_COLLECTION = "global_summaries"
MASKLETS_COLLECTION = "masklets"
AUDIO_EVENTS_COLLECTION = "audio_events"
VIDEO_METADATA_COLLECTION = "video_metadata"

# Vector dimensions
MEDIA_VECTOR_SIZE = settings.visual_embedding_dim
FACE_VECTOR_SIZE = 512  # InsightFace ArcFace
VOICE_VECTOR_SIZE = 256

# Text embedding dimension (model-dependent)
if "bge-m3" in SELECTED_MODEL.lower() or "mxbai" in SELECTED_MODEL.lower():
    TEXT_DIM = 1024
else:
    TEXT_DIM = settings.text_embedding_dim
