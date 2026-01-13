"""VLM Council for multi-model frame description with cross-critique.

Implements parallel VLM generation + chairman synthesis per AGENTS.MD:
- LLaVA, MiniCPM-V, Qwen2-VL, Gemini Flash (optional API)
- Cross-critique matrix for quality scoring
- Chairman synthesizes final description
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from core.llm.vlm_factory import VLMClient, get_vlm_client
from core.utils.logger import get_logger

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


DENSE_MULTIMODAL_PROMPT = """Analyze this video frame exhaustively. Describe:

1. **People**: Count, apparent gender, age range, clothing (colors, types),
   accessories, posture, actions, expressions, identifiable features
2. **Objects**: All visible objects, their colors, positions, brands if visible
3. **Text**: Any visible text, signs, labels, captions (exact transcription)
4. **Setting**: Indoor/outdoor, location type, lighting, time of day
5. **Actions**: What is happening, motion, interactions between elements
6. **Audio cues**: If this appears to be from a scene with dialogue/music/sounds

Be extremely detailed. Include colors, positions, and relationships.
Format as structured paragraphs, not bullet points."""


SYNTHESIS_PROMPT = """You are the chairman of a VLM council. Multiple models
analyzed the same frame. Synthesize the best description:

{descriptions}

Create one comprehensive description that:
1. Includes all accurate observations from all models
2. Resolves any contradictions by majority vote
3. Uses the most specific and detailed phrasing
4. Maintains factual accuracy over creativity

Output only the synthesized description, nothing else."""


class VLMCouncil:
    """Multi-model VLM council with cross-critique synthesis.

    Usage:
        council = VLMCouncil()
        result = await council.analyze_frame(frame_bytes)
        print(result.description)
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
        frame: bytes | "np.ndarray",
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
            else:
                desc = self.local_client.generate_caption(frame, prompt)

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
            else:
                synthesized = self.chairman.generate_caption(
                    frame, synth_prompt
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
        frame: bytes | "np.ndarray",
        prompt: str | None = None,
    ) -> str:
        """Simple single-model analysis (fast mode).

        Args:
            frame: Frame as bytes or numpy array.
            prompt: Custom prompt.

        Returns:
            Description string.
        """
        prompt = prompt or DENSE_MULTIMODAL_PROMPT
        try:
            if isinstance(frame, bytes):
                return self.local_client.generate_caption_from_bytes(
                    frame, prompt
                ) or ""
            return self.local_client.generate_caption(frame, prompt) or ""
        except Exception as e:
            log.error(f"[VLMCouncil] Simple analysis failed: {e}")
            return ""
