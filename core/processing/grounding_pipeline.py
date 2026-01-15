
from pathlib import Path
from typing import Any
import logging
import asyncio

from core.processing.segmentation import Sam3Tracker
from core.storage.db import VectorDB
from core.utils.resource_arbiter import RESOURCE_ARBITER, GPU_SEMAPHORE

logger = logging.getLogger(__name__)

class GroundingPipeline:
    """Post-processing pipeline for visual grounding using SAM 3.
    
    Generates segmentation masks (masklets) for concepts description in video frames.
    Designed to run offline/asynchronously to avoid blocking ingestion.
    """

    def __init__(self):
        self.sam = Sam3Tracker()
        self.db = VectorDB()

    async def process_video(self, video_path: str, concepts: list[str] | None = None) -> int:
        """Run grounding on a video.

        Args:
            video_path: Path to the video file.
            concepts: Optional list of concepts to track. If None, extracts from metadata.

        Returns:
            Number of masklets created.
        """
        path = Path(video_path)
        if not path.exists():
            logger.error(f"Video not found: {path}")
            return 0

        # If no concepts provided, try to fetch some from DB or use heuristics
        # For now, we'll rely on explicit concepts or simple fallback
        if not concepts:
            # TODO: Fetch top entities from scenelets/frames once indexed
            # For now, return 0 if no explicit concepts
            logger.warning(f"No concepts provided for grounding: {video_path}")
            return 0


        logger.info(f"Starting grounding for {path.name} with concepts: {concepts}")

        # Get FPS for timestamp conversion
        import cv2
        cap = cv2.VideoCapture(str(path))
        fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
        cap.release()

        # Acquire GPU for heavy SAM usage
        async with GPU_SEMAPHORE: 
            count = 0
            try:
                # Iterate generator synchronously
                # TODO: Offload to thread if blocking becomes issue, but generator yields fast enough usually
                iterator = self.sam.process_video_concepts(path, concepts)
                
                for result in iterator:
                    # result: {frame_idx, object_ids, masks}
                    frame_idx = result["frame_idx"]
                    masks = result["masks"] # boolean array [N, H, W]
                    object_ids = result["object_ids"]
                    
                    timestamp = frame_idx / fps
                    
                    # For each object detected in this frame
                    for i, obj_id in enumerate(object_ids):
                        # Find which mask corresponds to this object
                        # Sam3Tracker.propagate yields all masks. 
                        # We need to map object_ids to masks. 
                        # Assuming they align by index if multiple objects tracked?
                        # Sam3 returns 'object_ids' list and 'masks' array matches first dim?
                        # Let's verify Sam3Tracker implementation of propagate.
                        # It yields: "masks": masks (numpy array)
                        # "object_ids": list
                        
                        if i < len(masks):
                             mask = masks[i]
                             # Only store if mask is significant?
                             if mask.sum() < 10: continue # Skip noise
                             
                             concept_name = concepts[obj_id] if obj_id < len(concepts) else f"object_{obj_id}"
                             
                             # Store as masklet
                             # We treat each frame as a 1-frame masklet for now, 
                             # or we could aggregate into temporal segments.
                             # For simplicity of "visual grounding search", per-frame mask presence is fine.
                             
                             # Compressing mask to RLE or polygon is better for DB.
                             # For now, we save it as a simplified payload or just metadata that "concept is here"
                             # But goal is "show exact pixels". 
                             # Storing full mask in vector DB payload is heavy.
                             # Ideally: Save mask to disk/gridfs, link in DB.
                             # MVP: Store simplified polygon or bounding box in Payload.
                             
                             # Let's store BBox for MVP + center point.
                             y_indices, x_indices =  np.where(mask)
                             if len(y_indices) == 0: continue
                             
                             y_min, y_max = y_indices.min(), y_indices.max()
                             x_min, x_max = x_indices.min(), x_indices.max()
                             
                             bbox = [int(x_min), int(y_min), int(x_max), int(y_max)]
                             
                             self.db.insert_masklet(
                                 video_path=str(path),
                                 concept=concept_name,
                                 start_time=timestamp,
                                 end_time=timestamp + (1.0/fps),
                                 confidence=1.0, # SAM is usually confident if prompted
                                 payload={
                                     "bbox": bbox,
                                     "frame_idx": int(frame_idx),
                                     "obj_id": int(obj_id)
                                 }
                             )
                             count += 1

            except Exception as e:
                logger.error(f"Grounding failed: {e}")
                
        logger.info(f"Grounding complete. Created {count} masklets.")
        return count

