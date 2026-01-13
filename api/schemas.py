from __future__ import annotations

from pydantic import BaseModel, Field


class IngestRequest(BaseModel):
    path: str = ""
    encoded_path: str | None = None  # Base64-encoded path for Unicode preservation
    media_type_hint: str = "unknown"
    content_type_hint: str = "auto"
    start_time: float | None = None
    end_time: float | None = None


class ScanRequest(BaseModel):
    """Request body for folder scanning."""

    directory: str
    recursive: bool = True
    extensions: list[str] = Field(default=[".mp4", ".mkv", ".avi", ".mov", ".webm"])


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
    query: str
    use_rerank: bool = False
    limit: int = 20
    min_confidence: float = 0.0
    video_path: str | None = None
    person_filter: list[str] | None = None  # Filter by person names


class IdentityMergeRequest(BaseModel):
    target_identity_id: str


class IdentityRenameRequest(BaseModel):
    name: str


class RedactRequest(BaseModel):
    video_path: str
    identity_id: str
    output_path: str | None = None


class VoiceMergeRequest(BaseModel):
    target_speaker_id: str
    source_speaker_ids: list[str]


class FrameDescriptionRequest(BaseModel):
    description: str


class CreateClusterRequest(BaseModel):
    name: str = ""
    type: str = "manual"


class MoveFacesRequest(BaseModel):
    face_ids: list[str]
    target_cluster_id: str


class MergeClustersRequest(BaseModel):
    source_cluster_id: str | int
    target_cluster_id: str | int
    strategy: str = "merge_to_target"
    force: bool = False


class BulkApproveRequest(BaseModel):
    cluster_ids: list[str | int]


class AgentChatRequest(BaseModel):
    message: str
    use_tools: bool = True
    model: str = "llama3.2:3b"


class AgentToolRequest(BaseModel):
    arguments: dict = {}
