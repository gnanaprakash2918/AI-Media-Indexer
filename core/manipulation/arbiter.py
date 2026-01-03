"""Inpainting Arbiter: Decides optimal model based on scene dynamics."""

import cv2
import numpy as np
from pathlib import Path
from typing import Literal

from core.utils.logger import log


class InpaintingArbiter:
    """Analyzes scene to select best inpainting backend."""
    
    MOTION_THRESHOLD = 2.0
    
    def analyze_scene(
        self, 
        video_path: Path, 
        mask_path: Path | None = None,
        sample_frames: int = 30
    ) -> Literal["propainter", "wan"]:
        """Analyze video motion to decide inpainting backend.
        
        Args:
            video_path: Path to video file.
            mask_path: Optional path to mask video.
            sample_frames: Number of frames to analyze.
            
        Returns:
            'propainter' for static backgrounds, 'wan' for dynamic.
        """
        cap = cv2.VideoCapture(str(video_path))
        ret, prev_frame = cap.read()
        
        if not ret:
            cap.release()
            return "propainter"
            
        prev_gray = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)
        total_motion = 0.0
        frame_count = 0
        
        while frame_count < sample_frames:
            ret, frame = cap.read()
            if not ret:
                break
                
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            
            flow = cv2.calcOpticalFlowFarneback(
                prev_gray, gray, None,  # type: ignore
                0.5, 3, 15, 3, 5, 1.2, 0
            )
            mag, _ = cv2.cartToPolar(flow[..., 0], flow[..., 1])
            total_motion += np.mean(mag)
            
            prev_gray = gray
            frame_count += 1
            
        cap.release()
        
        avg_motion = total_motion / max(1, frame_count)
        log(f"[Arbiter] Average motion: {avg_motion:.4f}")
        
        if avg_motion < self.MOTION_THRESHOLD:
            log("[Arbiter] Selected: ProPainter (static background)")
            return "propainter"
        else:
            log("[Arbiter] Selected: Wan (dynamic background)")
            return "wan"
    
    def check_occlusion_recovery(
        self,
        video_path: Path,
        mask_frames: list[np.ndarray],
        threshold: float = 0.8
    ) -> bool:
        """Check if masked region is revealed in other frames.
        
        Args:
            video_path: Path to video.
            mask_frames: List of binary masks per frame.
            threshold: Fraction of frames where region must be visible.
            
        Returns:
            True if region is recoverable via propagation.
        """
        if not mask_frames:
            return True
            
        cap = cv2.VideoCapture(str(video_path))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        cap.release()
        
        masked_frames = len([m for m in mask_frames if np.any(m)])
        unmasked_ratio = 1.0 - (masked_frames / max(1, total_frames))
        
        return unmasked_ratio >= threshold
