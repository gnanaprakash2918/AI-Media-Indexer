"""Image analysis utilities that use an LLM to describe images.

The VisionAnalyzer provides a small async wrapper to construct prompts and
request image descriptions from the configured LLM. Supports both unstructured
descriptions and structured FrameAnalysis for FAANG-level search.
"""

import asyncio
from pathlib import Path
from typing import TYPE_CHECKING

from core.utils.logger import log
from core.utils.observe import observe
from llm.factory import LLMFactory
from llm.interface import LLMInterface

if TYPE_CHECKING:
    from core.knowledge.schemas import FrameAnalysis


# =============================================================================
# DENSE MULTIMODAL ANALYSIS PROMPT
# Captures EVERY MINUTE DETAIL for SOTA search quality
# =============================================================================

DENSE_MULTIMODAL_PROMPT = """Analyze this video frame and extract EVERY SINGLE DETAIL.
This is for a $289B investment-grade video search system. Accuracy is critical.

## EXTRACT ALL OF THE FOLLOWING:

### 1. PEOPLE (for EACH person visible)
- Position in frame (left/center/right, foreground/background)
- Estimated age range, gender
- Facial expression (happy, focused, surprised, etc.)
- Body posture (standing, sitting, walking, running, bent over)
- EACH clothing item:
  - Upper body: type, color, brand/logo if visible, pattern, material
  - Lower body: type, color, brand/logo if visible
  - Footwear: LEFT foot (type, color, brand), RIGHT foot (type, color, brand) - may differ!
  - Headwear: hat, glasses, spectacles (brand if visible like John Jacobs, Ray-Ban)
  - Accessories: watch, jewelry, bags (brand if visible like Balenciaga, Nike)
- Action being performed with PHYSICS details:
  - "throwing bowling ball with spin" not just "bowling"
  - "running at full speed" not just "moving"
  - Include result/outcome if visible ("hitting strike", "dropped ball")

### 2. VEHICLES (for EACH vehicle)
- Type (car, motorcycle, truck, bicycle)
- Brand and model if identifiable (Ferrari 488, Lamborghini Huracán, Tesla Model 3)
- Color (be specific: "cherry red", "matte black", "metallic silver")
- State (parked, moving, speeding, crashed)
- Position relative to other elements

### 3. OBJECTS
- Every significant object with:
  - Specific name (not "ball" but "bowling ball" or "cricket ball")
  - Color, material, size relative to scene
  - State (falling, stationary, spinning, broken)
  - Position (on table, in hand, on ground, in air)

### 4. TEXT/BRANDS (OCR)
- ALL readable text: signs, logos, labels, screens, clothing text
- Brand logos even if partially visible
- Numbers, scores, time displays

### 5. ENVIRONMENT
- Location type (bowling alley, restaurant, office, street, home)
- Specific venue name if visible ("Brunswick Lanes", "Starbucks")
- Indoor/outdoor
- Time of day (from lighting: morning, afternoon, evening, night)
- Weather if outdoor (sunny, cloudy, raining)
- Background elements

### 6. ACTIONS/EVENTS
- Primary action with cause and effect
- Secondary actions by other people/objects
- Temporal indicators ("about to", "just finished", "in progress")
- Motion blur or speed indicators
- Sequence position ("last pin falling", "first step")

### 7. AUDIO CUES (infer from visual)
- Likely sounds (pins crashing, engine roaring, crowd cheering)
- Music/ambient (party, quiet office, sports arena)
- Speech indicators (mouth open, gesturing while talking)

### 8. EMOTIONS/MOOD
- Expressions on faces
- Body language indicators
- Scene atmosphere (exciting, calm, tense, chaotic)

### 9. SPATIAL RELATIONSHIPS
- Who/what is next to whom/what
- Distances (close, far, touching)
- Interactions between elements

## OUTPUT FORMAT (JSON):
{
  "dense_caption": "<comprehensive description covering ALL above points>",
  "main_subject": "<primary focus of frame>",
  "people": [
    {
      "position": "<location in frame>",
      "description": "<age, gender, expression>",
      "clothing": {
        "upper": {"type": "", "color": "", "brand": ""},
        "lower": {"type": "", "color": ""},
        "left_foot": {"type": "", "color": "", "brand": ""},
        "right_foot": {"type": "", "color": "", "brand": ""},
        "accessories": []
      },
      "action": "<what they're doing with physics>",
      "action_result": "<outcome if visible>"
    }
  ],
  "vehicles": [{"type": "", "brand": "", "model": "", "color": "", "state": ""}],
  "objects": [{"name": "", "color": "", "state": "", "position": ""}],
  "visible_text": ["<all text/brands>"],
  "location": "<specific location>",
  "actions": ["<all actions>"],
  "temporal_hints": ["<timing indicators>"],
  "sounds_inferred": ["<likely sounds>"],
  "emotions": ["<emotions detected>"]
}

BE EXHAUSTIVE. Every detail matters for search accuracy."""

# Legacy prompt alias
STRUCTURED_ANALYSIS_PROMPT = DENSE_MULTIMODAL_PROMPT


class VisionAnalyzer:
    """Analyzing images using Multimodal LLMs.

    Supports two modes:
    1. describe() - Unstructured text description (legacy)
    2. analyze_frame() - Structured FrameAnalysis for search (preferred)
    
    LAZY LOADING: LLM is only loaded on first analyze call to prevent OOM.
    """

    def __init__(
        self,
        llm: LLMInterface | None = None,
        prompt_filename: str = "vision_prompt.txt",
    ):
        # LAZY LOADING: Store llm reference but don't create if not provided
        self._llm = llm
        self._llm_loaded = llm is not None
        self.prompt_filename = prompt_filename
        self.prompt: str | None = None
        log("[Vision] Initialized (lazy mode). LLM will load on first analyze call.")

    def _ensure_llm_loaded(self) -> None:
        """Lazy load LLM on first use."""
        if not self._llm_loaded:
            log("[Vision] Lazy loading LLM...")
            self._llm = LLMFactory.create_llm(provider="ollama")
            self._llm_loaded = True

        if self.prompt is None and self._llm is not None:
            try:
                self.prompt = self._llm.construct_user_prompt(self.prompt_filename)
                log(f"[Vision] Loaded prompt from {self.prompt_filename}")
            except FileNotFoundError:
                log("[Vision] Prompt file not found, using DENSE_MULTIMODAL_PROMPT")
                self.prompt = DENSE_MULTIMODAL_PROMPT

    @property
    def llm(self) -> LLMInterface:
        """Access LLM with lazy loading."""
        self._ensure_llm_loaded()
        assert self._llm is not None
        return self._llm


    @observe("vision_analyze_frame")
    async def analyze_frame(
        self,
        image_path: Path,
        *,
        identity_context: str | None = None,
        audio_context: str | None = None,
        temporal_context: str | None = None,
    ) -> "FrameAnalysis | None":
        """Analyze a frame and return structured FrameAnalysis for search.

        This is the preferred method for ingestion as it extracts:
        - Specific entity names (brands, food items, clothing)
        - Precise actions with physics
        - Visible text/OCR for brand matching
        - Scene context for location-based search

        Args:
            image_path: Path to the frame image.
            identity_context: Known face identities detected in frame (e.g., "Person in center: John, Person on left: Unknown")
            audio_context: Transcript from audio at this timestamp (e.g., "John: 'Let's go bowling'")
            temporal_context: Summary from previous frames (e.g., "[0.5s] Walking to bowling lane -> [1.0s] Picking up ball")

        Returns:
            FrameAnalysis object or None if analysis fails.
        """
        from core.knowledge.schemas import FrameAnalysis

        image_path = Path(image_path)
        if not image_path.exists() or not image_path.is_file():
            log(f"[Vision] Image not found: {image_path}")
            return None

        # Build enhanced prompt with multimodal context
        prompt_parts = [STRUCTURED_ANALYSIS_PROMPT]

        if identity_context:
            prompt_parts.append(f"\n\n## KNOWN IDENTITIES IN FRAME\n{identity_context}\nUse these names when describing the people.")

        if audio_context:
            prompt_parts.append(f"\n\n## AUDIO CONTEXT (what's being said)\n{audio_context}\nThis shows dialogue at this moment. Use it to understand the scene.")

        if temporal_context:
            prompt_parts.append(f"\n\n## PREVIOUS FRAMES (temporal context)\n{temporal_context}\nThis shows what happened before. Use it to understand continuing actions and narrative flow.")

        enhanced_prompt = "".join(prompt_parts)

        try:
            analysis = await self.llm.describe_image_structured(
                schema=FrameAnalysis,
                prompt=enhanced_prompt,
                image_path=image_path,
            )
            log(f"[Vision] Structured analysis: {analysis.action[:50] if analysis.action else 'no action'}...")
            return analysis
        except Exception as e:
            log(f"[Vision] Structured analysis failed for {image_path}: {e}")
            log(f"[Vision] Falling back to unstructured description for {image_path.name}")
            return None

    @observe("vision_describe")
    async def describe(self, image_path: Path, context: str | None = None) -> str:
        """Asynchronously describe an image using the configured LLM.

        Args:
            image_path: Path to the image file to describe.
            context: Optional context from previous frames to maintain narrative flow.

        Returns:
            A textual description produced by the LLM.

        Raises:
            FileNotFoundError: If the provided image file does not exist.
        """
        image_path = Path(image_path)
        if not image_path.exists() or not image_path.is_file():
            return f"Error: Image not found at {image_path}"

        self._ensure_llm_loaded()
        final_prompt = self.prompt
        if context:
            # Append context to help the model understand continuity
            final_prompt += f"\n\n[PREVIOUS FRAME CONTEXT]: {context}\nUse this context to infer continuing actions, but focus on the CURRENT image."

        return await self.llm.describe_image(
            prompt=final_prompt,
            image_path=image_path,
        )


if __name__ == "__main__":
    # Simple smoke test
    import sys

    if len(sys.argv) < 2:
        log("Usage: python -m core.processing.vision /path/to/image.jpg")
        raise SystemExit(1)

    img_path = Path(sys.argv[1])

    async def main() -> None:
        """CLI helper that calls the async describe method and prints result."""
        try:
            analyzer = VisionAnalyzer()
            description = await analyzer.describe(img_path)
            log(f"\n[Description]:\n{description}")
        except Exception as e:
            log(f"Vision tool failed: {e}")

    asyncio.run(main())
