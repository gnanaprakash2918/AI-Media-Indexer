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

            # Pipeline stores faces as: [{"bbox": [...], "cluster_id": X, "name": "...", "confidence": Y}]
            faces_data = frame.get("faces", [])
            for face in faces_data:
                overlays["faces"].append(
                    {
                        "timestamp": ts,
                        "bbox": face.get("bbox", []),
                        "label": face.get("name"),
                        "cluster_id": face.get("cluster_id"),
                        "confidence": face.get("confidence", 0),
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

            # Active speaker overlay - extract bboxes from faces_data
            if frame.get("is_active_speaker"):
                # Use bboxes from faces_data instead of undefined face_boxes
                face_bboxes = [f.get("bbox", []) for f in faces_data]
                for i, bbox in enumerate(face_bboxes):
                    if frame.get("speaking_face_idx") == i and bbox:
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

        # === VOICE DIARIZATION SEGMENTS ===
        # Get speaker segments for timeline visualization
        voice_segments = (
            pipeline.db.get_voice_segments_by_video(
                video_path=video_id,
                start_time=start_time,
                end_time=end_time,
            )
            if hasattr(pipeline.db, "get_voice_segments_by_video")
            else []
        )

        overlays["voice_diarization"] = []
        for segment in voice_segments:
            overlays["voice_diarization"].append(
                {
                    "start_time": segment.get("start_time", 0),
                    "end_time": segment.get("end_time", 0),
                    "speaker_label": segment.get("speaker_label", "Unknown"),
                    "speaker_name": segment.get("speaker_name"),
                    "voice_cluster_id": segment.get("voice_cluster_id", -1),
                    "color": "#8B5CF6",  # Purple for voice segments
                }
            )

        logger.info(
            f"[Overlays] {video_id}: {len(overlays['faces'])} faces, "
            f"{len(overlays['text_regions'])} text, {len(overlays['objects'])} objects, "
            f"{len(overlays.get('voice_diarization', []))} voice segments"
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


@router.get("/overlays/voice")
async def get_voice_overlays(
    video_id: Annotated[str, Query(description="Video path or ID")],
    pipeline: Annotated[IngestionPipeline, Depends(get_pipeline)],
    timestamp: Annotated[
        float | None, Query(description="Specific timestamp to check")
    ] = None,
) -> dict:
    """Get voice diarization overlays for a video.

    Returns speaker segments with their time ranges and speaker info.

    Args:
        video_id: Video path or identifier.
        pipeline: Ingestion pipeline instance.
        timestamp: Optional timestamp to filter around.

    Returns:
        Dict with voice segments and active speaker at timestamp.
    """
    if not pipeline or not pipeline.db:
        raise HTTPException(status_code=503, detail="Pipeline not initialized")

    try:
        # Get voice segments for this video
        segments = pipeline.db.get_voice_segments_for_media(video_id)

        # Format for overlay display
        voice_overlays = []
        active_speaker = None

        for seg in segments:
            start = seg.get("start", 0)
            end = seg.get("end", 0)
            speaker_label = seg.get("speaker_label", "Unknown")
            speaker_name = seg.get("speaker_name")

            overlay = {
                "type": "voice",
                "start_time": start,
                "end_time": end,
                "duration": end - start,
                "speaker_label": speaker_label,
                "speaker_name": speaker_name,
                "cluster_id": seg.get("voice_cluster_id", -1),
                "has_audio": bool(seg.get("audio_path")),
                "audio_url": seg.get("audio_path"),
                "color": _get_speaker_color(seg.get("voice_cluster_id", 0)),
            }
            voice_overlays.append(overlay)

            # Check if this speaker is active at the given timestamp
            if timestamp is not None and start <= timestamp <= end:
                active_speaker = {
                    "speaker_label": speaker_label,
                    "speaker_name": speaker_name or speaker_label,
                    "start_time": start,
                    "end_time": end,
                }

        return {
            "video_id": video_id,
            "timestamp": timestamp,
            "voice_segments": voice_overlays,
            "total_segments": len(voice_overlays),
            "active_speaker": active_speaker,
        }

    except Exception as e:
        logger.error(f"[Overlays] Voice lookup failed: {e}")
        raise HTTPException(status_code=500, detail=str(e)) from e


def _get_speaker_color(cluster_id: int) -> str:
    """Generate a consistent color for a speaker cluster."""
    colors = [
        "#EF4444",  # Red
        "#3B82F6",  # Blue
        "#10B981",  # Green
        "#F59E0B",  # Amber
        "#8B5CF6",  # Purple
        "#EC4899",  # Pink
        "#06B6D4",  # Cyan
        "#F97316",  # Orange
    ]
    if cluster_id < 0:
        return "#6B7280"  # Gray for unclustered
    return colors[cluster_id % len(colors)]
