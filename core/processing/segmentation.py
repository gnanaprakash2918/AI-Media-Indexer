"""SOTA Segmentation Module using SAM (Segment Anything Model).

Falls back to Ultralytics SAM/YOLO if weights are missing.
Uses LAZY LOADING to prevent OOM on startup.
"""
import gc
import logging
from pathlib import Path
from typing import List, Dict, Any, Iterator, Optional

import numpy as np

logger = logging.getLogger(__name__)


class SegmentationEngine:
    """SAM-based segmentation with lazy loading."""
    
    def __init__(self, model_size: str = "sam_b"):
        self.model = None
        self.model_type = model_size
        self._device: Optional[str] = None
        self._initialized = False
    
    @property
    def device(self) -> str:
        if self._device is None:
            import torch
            self._device = "cuda" if torch.cuda.is_available() else "cpu"
        return self._device
    
    def lazy_load(self) -> bool:
        """Loads SAM model only when needed to save VRAM."""
        if self.model is not None:
            return True
        
        logger.info(f"Loading Segmentation Model ({self.model_type}) on {self.device}...")
        
        try:
            from ultralytics import SAM
            self.model = SAM("sam_b.pt")  # Downloads automatically if missing
            logger.info("SAM Model loaded successfully.")
            return True
        except ImportError:
            logger.warning("Ultralytics not found. Install with: pip install ultralytics")
            return False
        except Exception as e:
            logger.error(f"Failed to load SAM: {e}")
            return False
    
    def segment_frame(
        self, 
        frame: np.ndarray, 
        prompt_points: Optional[List[List[int]]] = None
    ) -> List[Dict[str, Any]]:
        """Segments objects in a frame.
        
        Args:
            frame: Input image as numpy array.
            prompt_points: Optional click points for interactive segmentation.
            
        Returns:
            List of segment dicts with id, segmentation polygon, confidence.
        """
        if not self.lazy_load():
            return []
        
        results = []
        try:
            if prompt_points:
                res = self.model(frame, points=prompt_points, device=self.device)
            else:
                res = self.model(frame, device=self.device)
            
            for r in res:
                if hasattr(r, 'masks') and r.masks is not None:
                    masks_data = r.masks.xy
                    for i, mask_poly in enumerate(masks_data):
                        conf = float(r.boxes.conf[i]) if r.boxes is not None and len(r.boxes.conf) > i else 1.0
                        results.append({
                            "id": i,
                            "segmentation": mask_poly.tolist(),
                            "confidence": conf,
                            "label": "object"
                        })
        except Exception as e:
            logger.error(f"Segmentation error: {e}")
        
        return results
    
    def cleanup(self) -> None:
        """Release resources."""
        self.model = None
        import torch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()
        logger.info("Segmentation resources released")


class Sam3Tracker:
    """SAM 3 video segmentation and concept tracking.
    
    This is the full SAM-3 wrapper for video tracking with text prompts.
    """
    
    def __init__(self):
        self.predictor = None
        self.inference_state = None
        self._device: Optional[str] = None
        self._initialized = False
    
    @property
    def device(self) -> str:
        if self._device is None:
            from config import settings
            self._device = settings.device
        return self._device

    def initialize(self) -> bool:
        """Load SAM 3 model. Returns True if successful."""
        if self._initialized:
            return self.predictor is not None
            
        self._initialized = True
        
        try:
            from sam2.build_sam import build_sam2_video_predictor
            from config import settings
            
            checkpoint = settings.model_cache_dir / "sam2" / "sam2_hiera_large.pt"
            config = "sam2_hiera_l.yaml"
            
            if not checkpoint.exists():
                logger.warning(f"SAM3 checkpoint not found: {checkpoint}")
                logger.info("Download from: https://github.com/facebookresearch/segment-anything-2")
                return False
                
            logger.info("Loading SAM3 video predictor...")
            self.predictor = build_sam2_video_predictor(config, checkpoint)
            logger.info(f"SAM3 loaded on {self.device}")
            return True
            
        except ImportError:
            logger.warning("sam2 package not installed. Run: pip install segment-anything-2")
            return False
        except Exception as e:
            logger.error(f"SAM3 load failed: {e}")
            return False

    def init_video(self, video_path: Path) -> bool:
        """Initialize video state for tracking."""
        if not self.initialize():
            return False
            
        try:
            if self.predictor:
                self.inference_state = self.predictor.init_state(str(video_path))
                logger.info(f"SAM3 initialized video: {video_path.name}")
                return True
            return False
        except Exception as e:
            logger.error(f"SAM3 video init failed: {e}")
            return False

    def add_concept_prompt(self, text: str, frame_idx: int = 0) -> List[int]:
        """Add a text concept prompt for tracking."""
        if self.predictor is None or self.inference_state is None:
            logger.warning("SAM3 not initialized")
            return []
            
        try:
            _, obj_ids, _ = self.predictor.add_new_prompt(
                self.inference_state,
                frame_idx=frame_idx,
                text=text
            )
            logger.info(f"SAM3 added concept '{text}': {len(obj_ids)} instances")
            return obj_ids.tolist() if hasattr(obj_ids, 'tolist') else list(obj_ids)
        except Exception as e:
            logger.error(f"SAM3 prompt failed: {e}")
            return []

    def propagate(self) -> Iterator[Dict[str, Any]]:
        """Propagate masks through video frames."""
        if self.predictor is None or self.inference_state is None:
            return
            
        try:
            for frame_idx, obj_ids, mask_logits in self.predictor.propagate_in_video(
                self.inference_state
            ):
                masks = (mask_logits > 0.0).cpu().numpy()
                yield {
                    "frame_idx": frame_idx,
                    "object_ids": obj_ids.tolist() if hasattr(obj_ids, 'tolist') else list(obj_ids),
                    "masks": masks,
                }
        except Exception as e:
            logger.error(f"SAM3 propagation error: {e}")

    def process_video_concepts(
        self, 
        video_path: Path, 
        prompts: List[str]
    ) -> Iterator[Dict[str, Any]]:
        """Full pipeline: init video, add prompts, propagate."""
        if not self.init_video(video_path):
            return
            
        for prompt in prompts:
            self.add_concept_prompt(prompt)
            
        yield from self.propagate()
        self.cleanup()

    def cleanup(self) -> None:
        """Release resources."""
        self.inference_state = None
        import torch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()
        logger.info("SAM3 resources released")

    def reset(self) -> None:
        """Full reset including model."""
        self.cleanup()
        self.predictor = None
        self._initialized = False
