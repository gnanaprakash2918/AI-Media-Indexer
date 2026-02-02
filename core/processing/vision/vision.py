"""Image analysis utilities that use an LLM to describe images.

Prompts loaded from external files for easy customization.
"""

import asyncio
from pathlib import Path
from typing import TYPE_CHECKING

from core.utils.logger import log
from core.utils.observe import observe
from core.utils.prompt_loader import load_prompt
from llm.factory import LLMFactory
from llm.interface import LLMInterface

if TYPE_CHECKING:
    from core.knowledge.schemas import FrameAnalysis


# Load prompts from external files - NO HARDCODING
DENSE_MULTIMODAL_PROMPT = load_prompt("dense_multimodal")
STRUCTURED_ANALYSIS_PROMPT = DENSE_MULTIMODAL_PROMPT  # Legacy alias


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
    ) -> None:
        """Initializes the VisionAnalyzer in lazy mode.

        The LLM implementation is only initialized when analysis is first
        requested, which helps conserve VRAM during system startup.

        Args:
            llm: Optional pre-configured LLM interface.
            prompt_filename: The name of the file containing the vision prompt.
        """
        # LAZY LOADING: Store llm reference but don't create if not provided
        self._llm = llm
        self._llm_loaded = llm is not None
        self.prompt_filename = prompt_filename
        self.prompt: str | None = None

        # Register with Resource Arbiter
        try:
            from core.utils.resource_arbiter import RESOURCE_ARBITER

            # Register with a default VRAM estimate (e.g. 6GB for a 7B model)
            RESOURCE_ARBITER.register_model("vision_llm", self.unload_model)
        except ImportError:
            pass

        log(
            "[Vision] Initialized (lazy mode). LLM will load on first analyze call."
        )

    def unload_model(self) -> None:
        """Unload the LLM to free VRAM resources."""
        if self._llm is not None:
            self._llm = None
            self._llm_loaded = False

            import gc

            import torch

            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

            log("[Vision] LLM unloaded to free VRAM.")

    def _ensure_llm_loaded(self) -> None:
        """Loads the LLM and prompt template if they are not already cached."""
        if not self._llm_loaded:
            log("[Vision] Lazy loading LLM...")
            self._llm = LLMFactory.create_llm(provider="ollama")
            self._llm_loaded = True

        if self.prompt is None and self._llm is not None:
            try:
                self.prompt = self._llm.construct_user_prompt(
                    self.prompt_filename
                )
                log(f"[Vision] Loaded prompt from {self.prompt_filename}")
            except FileNotFoundError:
                log(
                    "[Vision] Prompt file not found, using DENSE_MULTIMODAL_PROMPT"
                )
                self.prompt = DENSE_MULTIMODAL_PROMPT

    @property
    def llm(self) -> LLMInterface:
        """Provides access to the lazy-loaded LLM interface.

        Returns:
            The initialized LLMInterface instance.
        """
        self._ensure_llm_loaded()
        assert self._llm is not None
        return self._llm

    @observe("vision_analyze_frame")
    async def analyze_frame(
        self,
        image_path: Path,
        *,
        video_context: str | None = None,
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
            video_context: Filename and metadata context to ground analysis (prevents VLM hallucinations).
            identity_context: Known face identities detected in frame.
            audio_context: Transcript from audio at this timestamp.
            temporal_context: Summary from previous frames.

        Returns:
            FrameAnalysis object or None if analysis fails.
        """
        from core.knowledge.schemas import FrameAnalysis

        image_path = Path(image_path)
        if not image_path.exists() or not image_path.is_file():
            log(f"[Vision] Image not found: {image_path}")
            return None

        # Build enhanced prompt with multimodal context
        # CRITICAL: Add video context FIRST to ground VLM and prevent hallucinations
        prompt_parts = []

        if video_context:
            prompt_parts.append(f"""## VIDEO CONTEXT (USE THIS TO GROUND YOUR ANALYSIS)
{video_context}

IMPORTANT RULES:
1. Base your description ONLY on what you SEE in the frame
2. Do NOT hallucinate conversations, events, or contexts not visible
3. If this is a song/music video, describe choreography/visuals - NOT imaginary conversations
4. If filename suggests content type (song, trailer, etc), use that to interpret ambiguous visuals
""")

        prompt_parts.append(STRUCTURED_ANALYSIS_PROMPT)

        if identity_context:
            prompt_parts.append(
                f"\n\n## KNOWN IDENTITIES IN FRAME\n{identity_context}\nUse these names when describing the people."
            )

        if audio_context:
            prompt_parts.append(
                f"\n\n## AUDIO CONTEXT (what's being said)\n{audio_context}\nThis shows dialogue at this moment. Use it to understand the scene."
            )

        if temporal_context:
            prompt_parts.append(
                f"\n\n## PREVIOUS FRAMES (temporal context)\n{temporal_context}\nThis shows what happened before. Use it to understand continuing actions and narrative flow."
            )

        enhanced_prompt = "\n".join(prompt_parts)

        # Retry logic for robustness against Ollama timeouts/transient errors
        max_retries = 3

        from core.utils.resource_arbiter import RESOURCE_ARBITER

        for attempt in range(max_retries):
            try:
                # Acquire VRAM -> this may trigger unloading of Whisper/other models
                async with RESOURCE_ARBITER.acquire("vision_llm", vram_gb=6.0):
                    analysis = await self.llm.describe_image_structured(
                        schema=FrameAnalysis,
                        prompt=enhanced_prompt,
                        image_path=image_path,
                    )
                log(
                    f"[Vision] Structured analysis: {analysis.action[:50] if analysis.action else 'no action'}..."
                )
                return analysis
            except Exception as e:
                wait_time = 2**attempt  # Exponential backoff: 1s, 2s, 4s
                if attempt < max_retries - 1:
                    log(
                        f"[Vision] Structured analysis failed (attempt {attempt + 1}/{max_retries}): {e}. Retrying in {wait_time}s..."
                    )
                    await asyncio.sleep(wait_time)
                else:
                    log(
                        f"[Vision] Structured analysis failed after {max_retries} attempts for {image_path}: {e}"
                    )
                    log(
                        f"[Vision] Falling back to unstructured description for {image_path.name}"
                    )
                    return None

    @observe("vision_describe")
    async def describe(
        self, image_path: Path, context: str | None = None
    ) -> str:
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
        final_prompt = self.prompt or ""
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
        log("Usage: python -m core.processing.vision.vision /path/to/image.jpg")
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

import numpy as np

class SaliencyDetector:
    """Visual saliency detection (Bing Video Search style).

    Identifies the most "important" regions of a frame
    for thumbnail selection and attention-based search.
    """

    def __init__(self):
        """Initialize saliency detector."""
        self._cv2 = None

    def _ensure_cv2(self) -> bool:
        """Ensure OpenCV is available."""
        if self._cv2 is not None:
            return True

        try:
            import cv2

            self._cv2 = cv2
            return True
        except ImportError:
            log.warning("[Saliency] OpenCV not available")
            return False

    async def compute_saliency(
        self,
        frame: np.ndarray,
    ) -> np.ndarray | None:
        """Compute saliency map for a frame.

        Args:
            frame: RGB frame as numpy array.

        Returns:
            Saliency map (same size as frame, grayscale).
        """
        if not self._ensure_cv2():
            return None

        try:
            cv2 = self._cv2

            # Use spectral residual saliency
            saliency = cv2.saliency.StaticSaliencySpectralResidual_create()

            # Compute
            success, saliency_map = saliency.computeSaliency(frame)

            if success:
                # Normalize to 0-255
                saliency_map = (saliency_map * 255).astype(np.uint8)
                return saliency_map

            return None

        except Exception as e:
            log.error(f"[Saliency] Computation failed: {e}")
            return None

    async def get_attention_region(
        self,
        frame: np.ndarray,
    ) -> dict[str, Any]:
        """Get the most salient region of the frame.

        Args:
            frame: RGB frame.

        Returns:
            Dict with bbox of attention region and score.
        """
        saliency_map = await self.compute_saliency(frame)

        if saliency_map is None:
            return {"bbox": None, "score": 0}

        try:
            cv2 = self._cv2

            # Threshold to find salient regions
            _, thresh = cv2.threshold(saliency_map, 127, 255, cv2.THRESH_BINARY)

            # Find contours
            contours, _ = cv2.findContours(
                thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
            )

            if not contours:
                return {"bbox": None, "score": 0}

            # Get largest contour
            largest = max(contours, key=cv2.contourArea)
            x, y, w, h = cv2.boundingRect(largest)

            # Calculate average saliency in this region
            region = saliency_map[y : y + h, x : x + w]
            score = float(region.mean()) / 255.0

            return {
                "bbox": [x, y, x + w, y + h],
                "score": round(score, 3),
            }

        except Exception as e:
            log.error(f"[Saliency] Region detection failed: {e}")
            return {"bbox": None, "score": 0}

    async def select_best_thumbnail(
        self,
        frames: list[np.ndarray],
    ) -> int:
        """Select the best frame for a thumbnail based on saliency.

        Args:
            frames: List of candidate frames.

        Returns:
            Index of the best frame.
        """
        if not frames:
            return 0

        best_idx = 0
        best_score = 0.0

        for i, frame in enumerate(frames):
            result = await self.get_attention_region(frame)
            score = result.get("score", 0)

            if score > best_score:
                best_score = score
                best_idx = i

        return best_idx

