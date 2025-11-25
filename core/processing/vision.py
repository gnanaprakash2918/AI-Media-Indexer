import asyncio
from pathlib import Path

from llm.factory import LLMFactory
from llm.interface import LLMInterface


class VisionAnalyzer:
    def __init__(
        self,
        llm: LLMInterface | None = None,
        prompt_filename: str = "vision_prompt.txt",
    ):
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
        """Async method to describe an image.
        Note: Removed asyncio.run() to prevent event loop conflicts.
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

    async def main():
        analyzer = VisionAnalyzer()
        description = await analyzer.describe(img_path)
        print(description)

    asyncio.run(main())
