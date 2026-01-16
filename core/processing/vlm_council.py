"""VLM Council for multi-model frame description.

Prompts loaded from external files.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING

from core.llm.vlm_factory import VLMClient, get_vlm_client
from core.utils.logger import get_logger
from core.utils.prompt_loader import load_prompt

if TYPE_CHECKING:
    import numpy as np

log = get_logger(__name__)


@dataclass
class VLMResponse:
    """Response from a single VLM model."""

    model_name: str
    description: str
    confidence: float = 1.0
    error: str | None = None


@dataclass
class CouncilResult:
    """Final synthesized result from VLM Council."""

    description: str
    confidence: float
    sources: list[VLMResponse] = field(default_factory=list)
    reasoning: str = ""


# Load prompts from external files
DENSE_MULTIMODAL_PROMPT = load_prompt("dense_multimodal")
SYNTHESIS_PROMPT = load_prompt("vlm_synthesis")


class VLMCouncil:
    """Multi-model VLM council with cross-critique synthesis.

    Usage:
        council = VLMCouncil()
        result = await council.analyze_frame(frame_bytes)
        print(result.description)

    Models are configured via COUNCIL_CONFIG.
    """

    def __init__(
        self,
        chairman_client: VLMClient | None = None,
        enable_api_models: bool = False,
    ):
        """Initialize VLM Council.

        Args:
            chairman_client: VLM client for synthesis (uses default if None).
            enable_api_models: Whether to use paid API models (Gemini, etc).
        """
        self._chairman = chairman_client
        self._local_client: VLMClient | None = None
        self._enable_api = enable_api_models

        # Log enabled models from config
        enabled = self.get_enabled_models()
        log.info(f"[VLMCouncil] Enabled models: {[m.name for m in enabled]}")

    def get_enabled_models(self) -> list:
        """Get enabled VLM models from council config.

        Returns:
            List of enabled ModelSpec objects.
        """
        try:
            from core.processing.council_config import COUNCIL_CONFIG

            return COUNCIL_CONFIG.get_enabled("vlm")
        except ImportError:
            return []

    @property
    def chairman(self) -> VLMClient:
        """Lazy load chairman VLM client."""
        if self._chairman is None:
            self._chairman = get_vlm_client()
        return self._chairman

    @property
    def local_client(self) -> VLMClient:
        """Lazy load local VLM client."""
        if self._local_client is None:
            self._local_client = get_vlm_client()
        return self._local_client

    async def analyze_frame(
        self,
        frame: bytes | np.ndarray,
        prompt: str | None = None,
        use_synthesis: bool = True,
    ) -> CouncilResult:
        """Analyze frame using VLM council.

        Args:
            frame: Frame as bytes or numpy array.
            prompt: Custom prompt (uses DENSE_MULTIMODAL_PROMPT if None).
            use_synthesis: Whether to synthesize responses.

        Returns:
            CouncilResult with description and sources.
        """
        prompt = prompt or DENSE_MULTIMODAL_PROMPT
        responses: list[VLMResponse] = []

        # Get response from local model
        try:
            if isinstance(frame, bytes):
                desc = self.local_client.generate_caption_from_bytes(
                    frame, prompt
                )
            elif hasattr(
                frame, "__array__"
            ):  # Check for numpy array like object
                try:
                    from typing import Any, cast

                    import cv2

                    # Use Mat type hint or cast to Any to avoid strict overload issues
                    # Pylance struggles with cv2.imencode overloads
                    f_arr = cast(Any, frame)
                    ret, buffer = cv2.imencode(".jpg", f_arr)
                    if not ret or buffer is None:
                        raise ValueError("Encoding failed")

                    desc = self.local_client.generate_caption_from_bytes(
                        buffer.tobytes(), prompt
                    )
                except Exception:
                    # Fallback if cv2 fails or not available, though unlikely for this project
                    log.warning("[VLMCouncil] Could not encode frame to bytes")
                    desc = ""
            else:
                # Assume path string
                desc = self.local_client.generate_caption(str(frame), prompt)

            responses.append(
                VLMResponse(
                    model_name="local_vlm",
                    description=desc or "",
                    confidence=1.0,
                )
            )
            log.debug(f"[VLMCouncil] Local VLM: {len(desc or '')} chars")
        except Exception as e:
            log.warning(f"[VLMCouncil] Local VLM failed: {e}")
            responses.append(
                VLMResponse(
                    model_name="local_vlm",
                    description="",
                    confidence=0.0,
                    error=str(e),
                )
            )

        # Filter successful responses
        valid = [r for r in responses if r.description and not r.error]

        if not valid:
            return CouncilResult(
                description="",
                confidence=0.0,
                sources=responses,
                reasoning="All VLM models failed",
            )

        # Single model - no synthesis needed
        if len(valid) == 1 or not use_synthesis:
            best = max(valid, key=lambda r: r.confidence)
            return CouncilResult(
                description=best.description,
                confidence=best.confidence,
                sources=responses,
                reasoning=f"Single model: {best.model_name}",
            )

        # Chairman synthesis for multiple responses
        try:
            desc_block = "\n\n---\n\n".join(
                f"**{r.model_name}**:\n{r.description}" for r in valid
            )
            synth_prompt = SYNTHESIS_PROMPT.format(descriptions=desc_block)

            if isinstance(frame, bytes):
                synthesized = self.chairman.generate_caption_from_bytes(
                    frame, synth_prompt
                )
            elif hasattr(frame, "__array__"):
                try:
                    from typing import Any, cast

                    import cv2

                    f_arr = cast(Any, frame)
                    ret, buffer = cv2.imencode(".jpg", f_arr)
                    if not ret:
                        raise ValueError("imencode failed")
                    synthesized = self.chairman.generate_caption_from_bytes(
                        buffer.tobytes(), synth_prompt
                    )
                except Exception:
                    synthesized = ""
            else:
                synthesized = self.chairman.generate_caption(
                    str(frame), synth_prompt
                )

            avg_conf = sum(r.confidence for r in valid) / len(valid)
            return CouncilResult(
                description=synthesized or valid[0].description,
                confidence=avg_conf,
                sources=responses,
                reasoning=f"Synthesized from {len(valid)} models",
            )

        except Exception as e:
            log.warning(f"[VLMCouncil] Synthesis failed: {e}")
            best = max(valid, key=lambda r: r.confidence)
            return CouncilResult(
                description=best.description,
                confidence=best.confidence * 0.9,
                sources=responses,
                reasoning=f"Synthesis failed, using {best.model_name}",
            )

    async def analyze_frame_simple(
        self,
        frame: "bytes | np.ndarray | str | Path",
        prompt: str | None = None,
    ) -> str:
        """Simple single-model analysis (fast mode).

        Args:
            frame: Frame as bytes, numpy array, or path.
            prompt: Custom prompt.

        Returns:
            Description string.
        """
        prompt = prompt or DENSE_MULTIMODAL_PROMPT
        try:
            if isinstance(frame, bytes):
                return (
                    self.local_client.generate_caption_from_bytes(frame, prompt)
                    or ""
                )

            if hasattr(frame, "__array__"):
                try:
                    from typing import Any, cast

                    import cv2

                    f_arr = cast(Any, frame)
                    ret, buffer = cv2.imencode(".jpg", f_arr)
                    if not ret or buffer is None:
                        return ""
                    return (
                        self.local_client.generate_caption_from_bytes(
                            buffer.tobytes(), prompt
                        )
                        or ""
                    )
                except Exception:
                    return ""

            # Assume path
            return self.local_client.generate_caption(str(frame), prompt) or ""
        except Exception as e:
            log.error(f"[VLMCouncil] Simple analysis failed: {e}")
            return ""
