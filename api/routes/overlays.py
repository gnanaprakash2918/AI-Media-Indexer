"""API routes for retrieving media overlays (faces, text, objects)."""
from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException, Query

from api.deps import get_pipeline
from core.ingestion.pipeline import IngestionPipeline
from core.utils.logger import logger

router = APIRouter()


@router.get("/overlays/{video_id}")
async def get_video_overlays(
    video_id: str,
    pipeline: Annotated[IngestionPipeline, Depends(get_pipeline)],
    start_time: Annotated[
        float | None, Query(description="Start time in seconds")
    ] = None,
    end_time: Annotated[
        float | None, Query(description="End time in seconds")
    ] = None,
):
    """Get all semantic overlays for a video within a time range.

    Args:
        video_id: Unique identifier or path of the video.
        pipeline: Ingestion pipeline instance.
        start_time: Optional start time filter.
        end_time: Optional end time filter.

    Returns:
        Dictionary containing faces, text regions, objects, and audio events.
    """
    if not pipeline or not pipeline.db:
        raise HTTPException(status_code=503, detail="Pipeline not initialized")

    try:
        overlays = {
            "video_id": video_id,
            "faces": [],
            "text_regions": [],
            "objects": [],
            "active_speakers": [],
            "loudness_events": [],
        }

        frames = (
            pipeline.db.get_frames_by_video(
                video_path=video_id,
                start_time=start_time,
                end_time=end_time,
            )
            if hasattr(pipeline.db, "get_frames_by_video")
            else []
        )

        for frame in frames:
            ts = frame.get("timestamp", 0)

            face_boxes = frame.get("face_boxes", [])
            for i, bbox in enumerate(face_boxes):
                overlays["faces"].append(
                    {
                        "timestamp": ts,
                        "bbox": bbox,
                        "label": frame.get("face_names", [])[i]
                        if i < len(frame.get("face_names", []))
                        else None,
                        "cluster_id": frame.get("face_cluster_ids", [])[i]
                        if i < len(frame.get("face_cluster_ids", []))
                        else None,
                        "color": "#22C55E",
                    }
                )

            ocr_boxes = frame.get("ocr_boxes", [])
            ocr_text = frame.get("ocr_text", "")
            if ocr_boxes:
                for bbox in ocr_boxes:
                    overlays["text_regions"].append(
                        {
                            "timestamp": ts,
                            "bbox": bbox,
                            "text": ocr_text,
                            "color": "#3B82F6",
                        }
                    )

            objects = frame.get("detected_objects", [])
            for obj in objects:
                overlays["objects"].append(
                    {
                        "timestamp": ts,
                        "bbox": obj.get("bbox", []),
                        "label": obj.get("label", ""),
                        "confidence": obj.get("confidence", 0),
                        "color": "#EF4444",
                    }
                )

            if frame.get("is_active_speaker"):
                for i, bbox in enumerate(face_boxes):
                    if frame.get("speaking_face_idx") == i:
                        overlays["active_speakers"].append(
                            {
                                "timestamp": ts,
                                "bbox": bbox,
                                "color": "#FBBF24",
                            }
                        )

        loudness_events = (
            pipeline.db.get_loudness_events(
                video_path=video_id,
                start_time=start_time,
                end_time=end_time,
            )
            if hasattr(pipeline.db, "get_loudness_events")
            else []
        )

        for event in loudness_events:
            overlays["loudness_events"].append(
                {
                    "timestamp": event.get("timestamp", 0),
                    "spl_db": event.get("spl_db", 0),
                    "lufs": event.get("lufs", 0),
                    "category": event.get("category", ""),
                }
            )

        logger.info(
            f"[Overlays] {video_id}: {len(overlays['faces'])} faces, {len(overlays['text_regions'])} text, {len(overlays['objects'])} objects"
        )
        return overlays

    except Exception as e:
        logger.error(f"[Overlays] Failed: {e}")
        raise HTTPException(status_code=500, detail=str(e)) from e


@router.get("/overlays/frame/{frame_id}")
async def get_frame_overlays(
    frame_id: str,
    pipeline: Annotated[IngestionPipeline, Depends(get_pipeline)],
):
    """Get all semantic overlays for a specific frame.

    Args:
        frame_id: Unique identifier for the frame.
        pipeline: Ingestion pipeline instance.

    Returns:
        Dictionary containing faces, text regions, and detected objects.
    """
    if not pipeline or not pipeline.db:
        raise HTTPException(status_code=503, detail="Pipeline not initialized")

    try:
        frame = (
            pipeline.db.get_frame_by_id(frame_id)
            if hasattr(pipeline.db, "get_frame_by_id")
            else None
        )
        if not frame:
            raise HTTPException(status_code=404, detail="Frame not found")

        overlays = {
            "frame_id": frame_id,
            "timestamp": frame.get("timestamp", 0),
            "faces": [],
            "text_regions": [],
            "objects": [],
            "clothing": [],
        }

        for i, bbox in enumerate(frame.get("face_boxes", [])):
            overlays["faces"].append(
                {
                    "bbox": bbox,
                    "label": frame.get("face_names", [])[i]
                    if i < len(frame.get("face_names", []))
                    else None,
                    "color": "#22C55E",
                }
            )

        for bbox in frame.get("ocr_boxes", []):
            overlays["text_regions"].append(
                {
                    "bbox": bbox,
                    "text": frame.get("ocr_text", ""),
                    "color": "#3B82F6",
                }
            )

        for obj in frame.get("detected_objects", []):
            overlays["objects"].append(
                {
                    "bbox": obj.get("bbox", []),
                    "label": obj.get("label", ""),
                    "color": "#EF4444",
                }
            )

        for clothing in frame.get("clothing_detections", []):
            overlays["clothing"].append(
                {
                    "bbox": clothing.get("bbox", []),
                    "description": clothing.get("description", ""),
                    "color": "#A855F7",
                }
            )

        return overlays

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"[Overlays] Frame lookup failed: {e}")
        raise HTTPException(status_code=500, detail=str(e)) from e
