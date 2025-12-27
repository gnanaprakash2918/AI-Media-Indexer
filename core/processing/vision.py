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


# Structured analysis prompt for high-quality entity extraction
STRUCTURED_ANALYSIS_PROMPT = """Analyze this video frame and extract structured information.

## CRITICAL RULES:
1. **SPECIFICITY**: Use exact names, not generic terms
   - "Idly" not "food", "Nike Air Jordan" not "shoes", "Tesla Model 3" not "car"
2. **BRANDS/TEXT**: Extract ALL visible text (signs, logos, labels)
3. **CLOTHING**: Describe with COLOR + TYPE (e.g., "blue Nike dri-fit t-shirt")
4. **ACTIONS**: Use precise verbs with physics (e.g., "bowling ball spinning toward pins")

## OUTPUT FORMAT (JSON):
Return a JSON object matching the FrameAnalysis schema with:
- main_subject: Primary person/object in focus
- action: Precise action occurring
- action_physics: Physical details (spinning, wobbling, falling slowly)
- entities: List of objects with {name, category, visual_details}
- scene: {location, action_narrative, visible_text[], cultural_context}

Focus on SEARCHABLE details that distinguish this frame from others."""


class VisionAnalyzer:
    """Analyzing images using Multimodal LLMs.

    Supports two modes:
    1. describe() - Unstructured text description (legacy)
    2. analyze_frame() - Structured FrameAnalysis for search (preferred)
    """

    def __init__(
        self,
        llm: LLMInterface | None = None,
        prompt_filename: str = "vision_prompt.txt",
    ):
        """Initialize the analyzer and load the prompt template.

        If the prompt template does not exist, a default template is used.
        """
        self.llm = llm or LLMFactory.create_llm(provider="ollama")
        self.prompt_filename = prompt_filename

        try:
            self.prompt = self.llm.construct_user_prompt(self.prompt_filename)
            log(f"[Vision] Loaded prompt from {self.prompt_filename}")
        except FileNotFoundError:
            log(f"[ERROR] Prompt file '{self.prompt_filename}' not found in prompts/.")
            raise

    @observe("vision_analyze_frame")
    async def analyze_frame(self, image_path: Path) -> "FrameAnalysis | None":
        """Analyze a frame and return structured FrameAnalysis for search.

        This is the preferred method for ingestion as it extracts:
        - Specific entity names (brands, food items, clothing)
        - Precise actions with physics
        - Visible text/OCR for brand matching
        - Scene context for location-based search

        Args:
            image_path: Path to the frame image.

        Returns:
            FrameAnalysis object or None if analysis fails.
        """
        from core.knowledge.schemas import FrameAnalysis

        image_path = Path(image_path)
        if not image_path.exists() or not image_path.is_file():
            log(f"[Vision] Image not found: {image_path}")
            return None

        try:
            analysis = await self.llm.describe_image_structured(
                schema=FrameAnalysis,
                prompt=STRUCTURED_ANALYSIS_PROMPT,
                image_path=image_path,
            )
            log(f"[Vision] Structured analysis: {analysis.action[:50] if analysis.action else 'no action'}...")
            return analysis
        except Exception as e:
            log(f"[Vision] Structured analysis failed: {e}, will fallback to describe()")
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
