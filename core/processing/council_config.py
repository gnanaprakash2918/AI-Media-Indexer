"""Pluggable council configuration for model selection.

Enables runtime selection of which models participate in each council:
- OSS-only mode: Local models only (Ollama, HuggingFace)
- Commercial-only: API models (Gemini, OpenAI, Anthropic)
- Combined: Best of both (default)

Usage:
    from core.processing.council_config import COUNCIL_CONFIG

    # Check which models are enabled
    vlm_models = COUNCIL_CONFIG.get_enabled("vlm")

    # Disable commercial models
    COUNCIL_CONFIG.set_mode("oss_only")
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from enum import Enum

from core.utils.logger import get_logger

log = get_logger(__name__)


class ModelType(Enum):
    """Model type classification."""

    OSS = "oss"  # Open source (local)
    COMMERCIAL = "commercial"  # Paid API
    HYBRID = "hybrid"  # Can be either


class CouncilMode(Enum):
    """Council operation mode."""

    OSS_ONLY = "oss_only"  # Local models only
    COMMERCIAL_ONLY = "commercial_only"  # API models only
    COMBINED = "combined"  # All available models
    CUSTOM = "custom"  # Per-model selection


@dataclass
class ModelSpec:
    """Specification for a council model."""

    name: str
    model_type: ModelType
    model_id: str  # HuggingFace ID or API model name
    enabled: bool = True
    weight: float = 1.0
    vram_gb: float = 0.0  # 0 for API models
    description: str = ""


@dataclass
class CouncilSpec:
    """Specification for a council."""

    name: str
    models: list[ModelSpec] = field(default_factory=list)
    min_models: int = 1  # Minimum models required
    voting_threshold: float = 0.5  # For consensus


# Default model configurations per council
DEFAULT_VLM_MODELS = [
    ModelSpec(
        name="llava",
        model_type=ModelType.OSS,
        model_id="llava:7b",
        vram_gb=6.0,
        description="LLaVA 7B - balanced quality/speed",
    ),
    ModelSpec(
        name="minicpm-v",
        model_type=ModelType.OSS,
        model_id="minicpm-v:8b",
        vram_gb=4.0,
        description="MiniCPM-V 8B - efficient vision",
    ),
    ModelSpec(
        name="qwen2-vl",
        model_type=ModelType.OSS,
        model_id="qwen2-vl:7b",
        vram_gb=6.0,
        description="Qwen2-VL 7B - strong vision",
    ),
    ModelSpec(
        name="gemini-flash",
        model_type=ModelType.COMMERCIAL,
        model_id="gemini-2.0-flash-exp",
        enabled=False,  # Disabled by default
        description="Gemini Flash - fast API",
    ),
    ModelSpec(
        name="gpt-4o",
        model_type=ModelType.COMMERCIAL,
        model_id="gpt-4o",
        enabled=False,
        description="GPT-4o - high quality API",
    ),
]

DEFAULT_ASR_MODELS = [
    ModelSpec(
        name="whisper-v3",
        model_type=ModelType.OSS,
        model_id="openai/whisper-large-v3",
        vram_gb=3.0,
        description="Whisper v3 - primary ASR",
    ),
    ModelSpec(
        name="whisper-turbo",
        model_type=ModelType.OSS,
        model_id="openai/whisper-large-v3-turbo",
        vram_gb=1.5,
        description="Whisper Turbo - fast ASR",
    ),
    ModelSpec(
        name="indic-conformer",
        model_type=ModelType.OSS,
        model_id="ai4bharat/indicconformer",
        vram_gb=2.0,
        description="IndicConformer - Indic languages",
    ),
    ModelSpec(
        name="seamless",
        model_type=ModelType.OSS,
        model_id="facebook/seamless-m4t-v2-large",
        vram_gb=4.0,
        description="SeamlessM4T - code-mixed",
    ),
]

DEFAULT_RERANK_MODELS = [
    ModelSpec(
        name="cross-encoder",
        model_type=ModelType.OSS,
        model_id="cross-encoder/ms-marco-MiniLM-L-6-v2",
        vram_gb=0.5,
        weight=0.35,
        description="Cross-encoder for text relevance",
    ),
    ModelSpec(
        name="bge-reranker",
        model_type=ModelType.OSS,
        model_id="BAAI/bge-reranker-v2-m3",
        vram_gb=1.0,
        weight=0.35,
        description="BGE Reranker v2",
    ),
    ModelSpec(
        name="vlm-reranker",
        model_type=ModelType.OSS,
        model_id="local_vlm",
        vram_gb=6.0,
        weight=0.30,
        description="VLM visual verification",
    ),
    ModelSpec(
        name="cohere-rerank",
        model_type=ModelType.COMMERCIAL,
        model_id="rerank-v3.5",
        enabled=False,
        weight=0.40,
        description="Cohere Rerank v3.5 API",
    ),
]

DEFAULT_AUDIO_EVENT_MODELS = [
    ModelSpec(
        name="clap",
        model_type=ModelType.OSS,
        model_id="laion/clap-htsat-unfused",
        vram_gb=1.0,
        description="LAION-CLAP zero-shot audio",
    ),
    ModelSpec(
        name="panns",
        model_type=ModelType.OSS,
        model_id="cnn14",
        vram_gb=0.5,
        enabled=False,  # Optional
        description="PANNs AudioSet classifier",
    ),
]


class CouncilConfig:
    """Central configuration for all councils.

    Supports runtime model selection and mode switching.
    """

    def __init__(self):
        """Initialize council configuration."""
        self.mode = CouncilMode.COMBINED
        self.councils: dict[str, CouncilSpec] = {
            "vlm": CouncilSpec(
                name="VLM Council",
                models=DEFAULT_VLM_MODELS.copy(),
            ),
            "asr": CouncilSpec(
                name="ASR Council",
                models=DEFAULT_ASR_MODELS.copy(),
            ),
            "rerank": CouncilSpec(
                name="Reranking Council",
                models=DEFAULT_RERANK_MODELS.copy(),
            ),
            "audio_event": CouncilSpec(
                name="Audio Event Council",
                models=DEFAULT_AUDIO_EVENT_MODELS.copy(),
            ),
        }
        self._callbacks: list[Callable] = []

    def set_mode(self, mode: str | CouncilMode) -> None:
        """Set council operation mode.

        Args:
            mode: 'oss_only', 'commercial_only', 'combined', or CouncilMode.
        """
        if isinstance(mode, str):
            mode = CouncilMode(mode)
        self.mode = mode

        # Update model enabled states
        for council in self.councils.values():
            for model in council.models:
                if mode == CouncilMode.OSS_ONLY:
                    model.enabled = model.model_type == ModelType.OSS
                elif mode == CouncilMode.COMMERCIAL_ONLY:
                    model.enabled = model.model_type == ModelType.COMMERCIAL
                elif mode == CouncilMode.COMBINED:
                    # Re-enable based on original defaults
                    model.enabled = True

        log.info(f"[CouncilConfig] Mode set to: {mode.value}")
        self._notify_change()

    def get_enabled(self, council: str) -> list[ModelSpec]:
        """Get enabled models for a council.

        Args:
            council: Council name ('vlm', 'asr', 'rerank', 'audio_event').

        Returns:
            List of enabled ModelSpec objects.
        """
        if council not in self.councils:
            return []
        return [m for m in self.councils[council].models if m.enabled]

    def set_model_enabled(
        self,
        council: str,
        model_name: str,
        enabled: bool,
    ) -> bool:
        """Enable or disable a specific model.

        Args:
            council: Council name.
            model_name: Model name within the council.
            enabled: Whether to enable the model.

        Returns:
            True if model was found and updated.
        """
        if council not in self.councils:
            return False

        for model in self.councils[council].models:
            if model.name == model_name:
                model.enabled = enabled
                log.info(
                    f"[CouncilConfig] {council}/{model_name} "
                    f"{'enabled' if enabled else 'disabled'}"
                )
                self._notify_change()
                return True
        return False

    def set_model_weight(
        self,
        council: str,
        model_name: str,
        weight: float,
    ) -> bool:
        """Set weight for a model in fusion.

        Args:
            council: Council name.
            model_name: Model name.
            weight: New weight (0.0 to 1.0).

        Returns:
            True if model was found and updated.
        """
        if council not in self.councils:
            return False

        for model in self.councils[council].models:
            if model.name == model_name:
                model.weight = max(0.0, min(1.0, weight))
                log.info(
                    f"[CouncilConfig] {council}/{model_name} "
                    f"weight set to {model.weight:.2f}"
                )
                self._notify_change()
                return True
        return False

    def add_model(self, council: str, model: ModelSpec) -> bool:
        """Add a new model to a council.

        Args:
            council: Council name.
            model: ModelSpec to add.

        Returns:
            True if model was added.
        """
        if council not in self.councils:
            return False

        # Check for duplicates
        for existing in self.councils[council].models:
            if existing.name == model.name:
                log.warning(
                    f"[CouncilConfig] Model {model.name} already exists"
                )
                return False

        self.councils[council].models.append(model)
        log.info(f"[CouncilConfig] Added {model.name} to {council}")
        self._notify_change()
        return True

    def to_dict(self) -> dict:
        """Export configuration as dictionary (for API/frontend).

        Returns:
            Configuration as nested dict.
        """
        return {
            "mode": self.mode.value,
            "councils": {
                name: {
                    "name": spec.name,
                    "min_models": spec.min_models,
                    "models": [
                        {
                            "name": m.name,
                            "type": m.model_type.value,
                            "model_id": m.model_id,
                            "enabled": m.enabled,
                            "weight": m.weight,
                            "vram_gb": m.vram_gb,
                            "description": m.description,
                        }
                        for m in spec.models
                    ],
                }
                for name, spec in self.councils.items()
            },
        }

    def from_dict(self, config: dict) -> None:
        """Import configuration from dictionary.

        Args:
            config: Configuration dict from to_dict().
        """
        if "mode" in config:
            self.set_mode(config["mode"])

        if "councils" in config:
            for council_name, council_data in config["councils"].items():
                if council_name not in self.councils:
                    continue
                for model_data in council_data.get("models", []):
                    name = model_data.get("name")
                    if "enabled" in model_data:
                        self.set_model_enabled(
                            council_name, name, model_data["enabled"]
                        )
                    if "weight" in model_data:
                        self.set_model_weight(
                            council_name, name, model_data["weight"]
                        )

    def on_change(self, callback: Callable) -> None:
        """Register callback for configuration changes.

        Args:
            callback: Function to call on changes.
        """
        self._callbacks.append(callback)

    def _notify_change(self) -> None:
        """Notify all registered callbacks of a change."""
        for callback in self._callbacks:
            try:
                callback()
            except Exception as e:
                log.error(f"[CouncilConfig] Callback error: {e}")


# Global singleton
COUNCIL_CONFIG = CouncilConfig()
