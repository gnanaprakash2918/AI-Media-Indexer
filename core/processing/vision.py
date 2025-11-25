"""Image analysis utilities that use an LLM to describe images.

The VisionAnalyzer provides a small async wrapper to construct prompts and
request image descriptions from the configured LLM.
"""

import asyncio
from pathlib import Path

from llm.factory import LLMFactory
from llm.interface import LLMInterface


class VisionAnalyzer:
    """Helper to create descriptive text for an image for indexing."""

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
        except FileNotFoundError:
            self.prompt = (
                "Describe this scene in detailed, search-friendly language. "
                "Identify objects, colors, positions, background context, "
                "actions, and visible text. Produce concise but rich output "
                "optimized for search and indexing."
            )

    async def describe(self, image_path: Path) -> str:
        """Asynchronously describe an image using the configured LLM.

        Args:
            image_path: Path to the image file to describe.

        Returns:
            A textual description produced by the LLM.

        Raises:
            FileNotFoundError: If the provided image file does not exist.
        """
        image_path = Path(image_path)
        if not image_path.exists() or not image_path.is_file():
            raise FileNotFoundError(f"Image not found: {image_path}")

        return await self.llm.describe_image(
            prompt=self.prompt,
            image_path=image_path,
        )


if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        print("Usage: python -m core.processing.vision /path/to/image.jpg")
        raise SystemExit(1)

    img_path = Path(sys.argv[1])

    async def main() -> None:
        """CLI helper that calls the async describe method and prints result."""
        analyzer = VisionAnalyzer()
        description = await analyzer.describe(img_path)
        print(description)

    asyncio.run(main())
