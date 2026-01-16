"""API request and response schemas using Pydantic."""

from __future__ import annotations

from pydantic import BaseModel, Field


class IngestRequest(BaseModel):
    """Configuration for starting a media ingestion job."""

    path: str = ""
    encoded_path: str | None = (
        None  # Base64-encoded path for Unicode preservation
    )
    media_type_hint: str = "unknown"
    content_type_hint: str = "auto"
    start_time: float | None = None
    end_time: float | None = None


class ScanRequest(BaseModel):
    """Request body for folder scanning."""

    directory: str
    recursive: bool = True
    extensions: list[str] = Field(
        default=[".mp4", ".mkv", ".avi", ".mov", ".webm"]
    )


class ConfigUpdate(BaseModel):
    """Configuration update model."""

    device: str | None = None
    compute_type: str | None = None
    frame_interval: int | None = None
    frame_sample_ratio: int | None = None
    face_detection_threshold: float | None = None
    face_detection_resolution: int | None = None
    language: str | None = None
    llm_provider: str | None = None
    ollama_base_url: str | None = None
    ollama_model: str | None = None
    google_api_key: str | None = None
    hf_token: str | None = None
    enable_voice_analysis: bool | None = None
    enable_resource_monitoring: bool | None = None


class NameFaceRequest(BaseModel):
    """Request body for naming a face cluster."""

    name: str


class AdvancedSearchRequest(BaseModel):
    """Aggregated search request with multi-modal filters and reranking."""

    query: str
    use_rerank: bool = False
    limit: int = 20
    min_confidence: float = 0.0
    video_path: str | None = None
    person_filter: list[str] | None = None  # Filter by person names
    face_cluster_id: int | None = None  # Filter by specific face cluster ID


class IdentityMergeRequest(BaseModel):
    """Parameters for merging two identity clusters."""

    target_identity_id: str


class IdentityRenameRequest(BaseModel):
    """Parameters for renaming an existing identity."""

    name: str


class RedactRequest(BaseModel):
    """Parameters for video redaction based on identity."""

    video_path: str
    identity_id: str
    output_path: str | None = None


class VoiceMergeRequest(BaseModel):
    """Parameters for merging speaker voice clusters."""

    target_speaker_id: str
    source_speaker_ids: list[str]


class FrameDescriptionRequest(BaseModel):
    """Custom description update for a specific frame."""

    description: str


class CreateClusterRequest(BaseModel):
    """Configuration for manual identity cluster creation."""

    name: str = ""
    type: str = "manual"


class MoveFacesRequest(BaseModel):
    """Parameters for reassignment of face points between clusters."""

    face_ids: list[str]
    target_cluster_id: str


class MergeClustersRequest(BaseModel):
    """Parameters for cluster-level merge operations."""

    source_cluster_id: str | int
    target_cluster_id: str | int
    strategy: str = "merge_to_target"
    force: bool = False


class BulkApproveRequest(BaseModel):
    """Parameters for batch approval of identity clusters."""

    cluster_ids: list[str | int]


class AgentChatRequest(BaseModel):
    """Parameters for initiating a conversation with the AI agent."""

    message: str
    use_tools: bool = True
    model: str = "llama3.2:3b"


class AgentToolRequest(BaseModel):
    """Parameters for manual tool execution by the agent."""

    arguments: dict = {}
