"""Image analysis utilities that use an LLM to describe images.

The VisionAnalyzer provides a small async wrapper to construct prompts and
request image descriptions from the configured LLM.
"""

import asyncio
from pathlib import Path

from core.utils.logger import log
from core.utils.observe import observe
from llm.factory import LLMFactory
from llm.interface import LLMInterface


class VisionAnalyzer:
    """Analyzing images using Multimodal LLMs.

    Acts as a 'Tool' in the Agentic workflow.
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
