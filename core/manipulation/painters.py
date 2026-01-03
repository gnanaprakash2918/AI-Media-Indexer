"""Wrappers for video inpainting engines."""

import shutil
from pathlib import Path
from abc import ABC, abstractmethod

import cv2
import numpy as np

from core.utils.logger import log


class ManipulationEngine(ABC):
    """Base class for video manipulation engines."""
    
    @abstractmethod
    def inpaint(
        self, 
        video_path: Path, 
        mask_data: dict, 
        output_path: Path
    ) -> bool:
        """Perform inpainting on video.
        
        Args:
            video_path: Input video path.
            mask_data: Dict with mask frames or mask video path.
            output_path: Where to save result.
            
        Returns:
            True if successful.
        """
        pass


class ProPainterEngine(ManipulationEngine):
    """Propagation-based inpainting using ProPainter."""
    
    def __init__(self):
        self.model = None
        
    def load_model(self) -> bool:
        try:
            log("[ProPainter] Loading model...")
            # Placeholder for actual ProPainter import
            # from propainter import ProPainter
            self.model = True
            return True
        except ImportError:
            log("[ProPainter] Not installed. Run: pip install propainter")
            return False
    
    def inpaint(
        self, 
        video_path: Path, 
        mask_data: dict, 
        output_path: Path
    ) -> bool:
        if not self.load_model():
            return False
            
        log(f"[ProPainter] Processing {video_path.name}...")
        
        try:
            # Placeholder - actual implementation would call ProPainter inference
            shutil.copy(video_path, output_path)
            log(f"[ProPainter] Output: {output_path}")
            return True
        except Exception as e:
            log(f"[ProPainter] Failed: {e}")
            return False


class WanVideoEngine(ManipulationEngine):
    """Generative inpainting using Wan 2.1/2.2."""
    
    def __init__(self):
        self.model = None
        
    def load_model(self) -> bool:
        try:
            log("[Wan] Loading model...")
            # Placeholder for actual Wan import
            self.model = True
            return True
        except ImportError:
            log("[Wan] Not installed")
            return False
    
    def inpaint(
        self, 
        video_path: Path, 
        mask_data: dict, 
        output_path: Path
    ) -> bool:
        if not self.load_model():
            return False
            
        log(f"[Wan] Processing {video_path.name}...")
        
        try:
            # Placeholder - actual implementation would call Wan inference
            # with FlowEdit guidance and negative prompts
            shutil.copy(video_path, output_path)
            log(f"[Wan] Output: {output_path}")
            return True
        except Exception as e:
            log(f"[Wan] Failed: {e}")
            return False


class PrivacyBlur:
    """Lightweight GPU-accelerated blurring for privacy redaction."""
    
    def __init__(self, blur_type: str = "gaussian", kernel_size: int = 51):
        self.blur_type = blur_type
        self.kernel_size = kernel_size
        
    def apply(
        self,
        frame: "np.ndarray",
        mask: "np.ndarray"
    ) -> "np.ndarray":
        """Apply blur to masked region of frame.
        
        Args:
            frame: BGR image.
            mask: Binary mask of region to blur.
            
        Returns:
            Frame with blurred region.
        """
        import cv2
        import numpy as np
        
        if self.blur_type == "gaussian":
            blurred = cv2.GaussianBlur(frame, (self.kernel_size, self.kernel_size), 0)
        elif self.blur_type == "pixelate":
            h, w = frame.shape[:2]
            small = cv2.resize(frame, (w // 16, h // 16), interpolation=cv2.INTER_LINEAR)
            blurred = cv2.resize(small, (w, h), interpolation=cv2.INTER_NEAREST)
        else:
            blurred = cv2.blur(frame, (self.kernel_size, self.kernel_size))
            
        mask_3ch = np.stack([mask] * 3, axis=-1) if mask.ndim == 2 else mask
        result = np.where(mask_3ch > 0, blurred, frame)
        
        return result.astype(np.uint8)
