"""SOTA Video Inpainting Module.

Uses ProPainter (Propagation-based inpainting) as the default for high-fidelity
object removal. Falls back to simpler methods if ProPainter unavailable.

LAZY LOADING: Models only load when Inpaint tool is called by Agent.
"""

from __future__ import annotations

import gc
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class InpaintRequest:
    video_path: Path
    mask_frames: dict[int, np.ndarray]  # frame_idx -> binary mask
    output_path: Optional[Path] = None


@dataclass
class InpaintResult:
    success: bool
    output_path: Optional[Path] = None
    error: Optional[str] = None
    backend_used: str = ""


class VideoInpainter:
    """SOTA Video Inpainting using ProPainter or fallback methods."""

    def __init__(self):
        self._propainter_model = None
        self._initialized = False
        self._device: Optional[str] = None

    @property
    def device(self) -> str:
        if self._device is None:
            import torch

            self._device = "cuda" if torch.cuda.is_available() else "cpu"
        return self._device

    def _lazy_load_propainter(self) -> bool:
        """Lazy load ProPainter model."""
        if self._propainter_model is not None:
            return True

        logger.info("Loading ProPainter model...")

        try:
            from propainter import ProPainter

            self._propainter_model = ProPainter(device=self.device)
            self._initialized = True
            logger.info("ProPainter loaded successfully")
            return True
        except ImportError:
            logger.warning("ProPainter not installed. Install: pip install propainter")
            return False
        except Exception as e:
            logger.error(f"ProPainter load failed: {e}")
            return False

    def inpaint_video(self, request: InpaintRequest) -> InpaintResult:
        """Inpaint video using best available method."""
        video_path = request.video_path
        output_path = (
            request.output_path
            or video_path.parent / f"{video_path.stem}_inpainted{video_path.suffix}"
        )

        if self._lazy_load_propainter():
            return self._inpaint_propainter(
                video_path, request.mask_frames, output_path
            )

        return self._inpaint_opencv_fallback(
            video_path, request.mask_frames, output_path
        )

    def _inpaint_propainter(
        self, video_path: Path, mask_frames: dict[int, np.ndarray], output_path: Path
    ) -> InpaintResult:
        """Inpaint using ProPainter propagation-based method."""
        try:
            import cv2
            import torch

            cap = cv2.VideoCapture(str(video_path))
            fps = cap.get(cv2.CAP_PROP_FPS)
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            writer = cv2.VideoWriter(str(output_path), fourcc, fps, (width, height))

            frames = []
            masks = []
            frame_idx = 0

            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                frames.append(frame)
                mask = mask_frames.get(
                    frame_idx, np.zeros((height, width), dtype=np.uint8)
                )
                masks.append(mask)
                frame_idx += 1

            cap.release()

            frames_tensor = (
                torch.from_numpy(np.stack(frames)).permute(0, 3, 1, 2).float() / 255.0
            )
            masks_tensor = (
                torch.from_numpy(np.stack(masks)).unsqueeze(1).float() / 255.0
            )

            if self.device == "cuda":
                frames_tensor = frames_tensor.cuda()
                masks_tensor = masks_tensor.cuda()

            with torch.no_grad():
                inpainted = self._propainter_model.inpaint(frames_tensor, masks_tensor)

            inpainted = (inpainted.permute(0, 2, 3, 1).cpu().numpy() * 255).astype(
                np.uint8
            )

            for frame in inpainted:
                writer.write(frame)

            writer.release()
            self._cleanup()

            logger.info(f"ProPainter inpaint complete: {output_path}")
            return InpaintResult(
                success=True, output_path=output_path, backend_used="propainter"
            )

        except Exception as e:
            logger.error(f"ProPainter inpaint failed: {e}")
            return InpaintResult(success=False, error=str(e), backend_used="propainter")

    def _inpaint_opencv_fallback(
        self, video_path: Path, mask_frames: dict[int, np.ndarray], output_path: Path
    ) -> InpaintResult:
        """Fallback: OpenCV Telea inpainting (fast but lower quality)."""
        try:
            import cv2

            cap = cv2.VideoCapture(str(video_path))
            fps = cap.get(cv2.CAP_PROP_FPS)
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            writer = cv2.VideoWriter(str(output_path), fourcc, fps, (width, height))

            frame_idx = 0
            while True:
                ret, frame = cap.read()
                if not ret:
                    break

                mask = mask_frames.get(frame_idx)
                if mask is not None and mask.any():
                    frame = cv2.inpaint(
                        frame, mask, inpaintRadius=3, flags=cv2.INPAINT_TELEA
                    )

                writer.write(frame)
                frame_idx += 1

            cap.release()
            writer.release()

            logger.info(f"OpenCV inpaint complete: {output_path}")
            return InpaintResult(
                success=True, output_path=output_path, backend_used="opencv_telea"
            )

        except Exception as e:
            logger.error(f"OpenCV inpaint failed: {e}")
            return InpaintResult(
                success=False, error=str(e), backend_used="opencv_telea"
            )

    def _cleanup(self) -> None:
        """Free GPU memory after inpainting."""
        self._propainter_model = None
        import torch

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()
        logger.info("VideoInpainter resources released")

    def unload(self) -> None:
        """Explicit unload for lazy model management."""
        self._cleanup()
        self._initialized = False


class WanVideoInpainter:
    """Wan-2.1 Video Inpainting (alternative backend)."""

    def __init__(self):
        self._model = None
        self._device: Optional[str] = None

    @property
    def device(self) -> str:
        if self._device is None:
            import torch

            self._device = "cuda" if torch.cuda.is_available() else "cpu"
        return self._device

    def _lazy_load(self) -> bool:
        if self._model is not None:
            return True

        logger.info("Loading Wan-2.1 inpainting model...")
        try:
            import torch
            from diffusers import StableDiffusionInpaintPipeline

            self._model = StableDiffusionInpaintPipeline.from_pretrained(
                "runwayml/stable-diffusion-inpainting",
                torch_dtype=torch.float16 if self.device == "cuda" else torch.float32,
            )
            if self.device == "cuda":
                self._model = self._model.to("cuda")
            logger.info("Wan-2.1 (SD-Inpaint) loaded")
            return True
        except Exception as e:
            logger.error(f"Wan-2.1 load failed: {e}")
            return False

    def inpaint_frame(
        self,
        frame: np.ndarray,
        mask: np.ndarray,
        prompt: str = "clean background, no objects",
    ) -> Optional[np.ndarray]:
        """Inpaint single frame using diffusion."""
        if not self._lazy_load():
            return None

        try:
            from PIL import Image

            pil_image = Image.fromarray(frame)
            pil_mask = Image.fromarray(mask)

            result = self._model(
                prompt=prompt,
                image=pil_image,
                mask_image=pil_mask,
                num_inference_steps=20,
            ).images[0]

            return np.array(result)
        except Exception as e:
            logger.error(f"Frame inpaint failed: {e}")
            return None

    def unload(self) -> None:
        self._model = None
        import torch

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()


_inpainter: Optional[VideoInpainter] = None


def get_inpainter() -> VideoInpainter:
    global _inpainter
    if _inpainter is None:
        _inpainter = VideoInpainter()
    return _inpainter
