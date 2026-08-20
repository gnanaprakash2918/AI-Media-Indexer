from enum import Enum
from pydantic import Field, SecretStr, computed_field

from .hardware import _HW_PROFILE


class LLMProvider(str, Enum):
    """Supported LLM providers."""
    VLLM = "vllm"
    GEMINI = "gemini"
    OLLAMA = "ollama"


class LLMSettings:
    """LLM and VLM configuration."""
    
    agent_model: str = Field(default="llama3.1", description="Model for Agent CLI")
    llm_provider: LLMProvider = Field(default=LLMProvider.VLLM, description="LLM provider for all text/vision inference.")

    vllm_base_url: str = Field(
        default="http://localhost:8000",
        validation_alias="VLLM_BASE_URL",
        description="Base URL for the vLLM (or any OpenAI-compatible) endpoint",
    )
    vllm_api_key: str | None = Field(
        default=None,
        validation_alias="VLLM_API_KEY",
        description="API key for vLLM endpoint",
    )
    vlm_model_id: str = Field(
        default="Qwen3-VL-2B-Instruct",
        validation_alias="VLM_MODEL_ID",
        description="Logical model identifier",
    )
    vlm_endpoint_model_name: str = Field(
        default="Qwen/Qwen3-VL-2B-Instruct",
        validation_alias="VLM_ENDPOINT_MODEL_NAME",
        description="Model name sent to the vLLM endpoint",
    )

    ollama_base_url: str = Field(default="http://localhost:11434", validation_alias="OLLAMA_BASE_URL")
    ollama_vision_model: str = Field(default="llava:7b", validation_alias="OLLAMA_VISION_MODEL")
    ollama_text_model: str = Field(default="llama3.1", validation_alias="OLLAMA_TEXT_MODEL")

    gemini_api_key: SecretStr | None = Field(default=None, validation_alias="GOOGLE_API_KEY")
    gemini_model: str = "gemini-2.5-flash"

    tmdb_api_key: str | None = None
    omdb_api_key: str | None = None
    brave_api_key: str | None = Field(default=None, description="Brave Search API key")
    enable_external_search: bool = Field(default=False)
    hf_token: str | None = Field(default=None, validation_alias="HF_TOKEN")

    ai_provider_vision: str = Field(default="vllm")
    ai_provider_text: str = Field(default="vllm")

    vlm_max_frames: int = Field(default=32)
    vlm_max_tokens: int = Field(default=128)
    vlm_concurrency: int = Field(
        default=_HW_PROFILE["worker_count"] * 2,
        description="Max concurrent VLM calls.",
    )
    enable_frame_vlm: bool = Field(default=True)
    enable_hybrid_vlm: bool = Field(default=True)

    embedding_model_override: str = Field(default="")
    text_embedding_dim: int = Field(default=4096)
    visual_embedding_dim: int = Field(default=1152)
    siglip_model: str = Field(default="google/siglip-so400m-patch14-384")
    enable_visual_embeddings: bool = Field(default=True)
    video_embedding_dim: int = Field(default=1024)
    enable_video_embeddings: bool = Field(default=True)
    visual_features_dim: int = Field(default=1152)
    visual_encoder_type: str = Field(default="siglip")
    visual_encoder_fallback: bool = Field(default=True)

    @computed_field
    @property
    def video_vlm_model_id(self) -> str:
        """Back-compat alias for vlm_endpoint_model_name."""
        return self.vlm_endpoint_model_name

    @computed_field
    @property
    def effective_embedding_model(self) -> str:
        """ALWAYS use SOTA embedding model. Never downgrade for quality."""
        return self.embedding_model_override
