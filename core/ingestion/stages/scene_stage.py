"""Scene detection, aggregation, and VLM captioning stage.

Extracted from IngestionPipeline to reduce God class size.
IngestionPipeline inherits from SceneStageMixin to compose these methods.
"""

from __future__ import annotations

import hashlib
import uuid
from pathlib import Path
from typing import TYPE_CHECKING, Any

import cv2
import numpy as np

from config import settings
from core.llm.vlm_factory import get_vlm_client
from core.processing.scene_detector import detect_scenes, extract_scene_frame
from core.storage.db import VectorDB
from core.utils.logger import logger
from core.utils.progress import progress_tracker

if TYPE_CHECKING:
    pass


class SceneStageMixin:
    """Scene detection, aggregation, and VLM captioning stage."""

    # These will be available via IngestionPipeline inheritance
    db: VectorDB

    # Type stub for type checkers (allows accessing self.* in mixin)
    def __getattr__(self, name: str) -> Any: ...

    async def _process_scene_captions(
        self,
        path: Path,
        job_id: str | None = None,
        chunk_start: float | None = None,
        chunk_end: float | None = None,
    ) -> None:
        """Processes scene boundaries and aggregates multi-modal data.

        Identifies scene changes, creates visual summaries for each scene
        using VLM, aggregates dialogue and frame-level entities/faces,
        and stores everything with multi-vector embeddings for hybrid search.

        Args:
            path: Path to the media file.
            job_id: Optional ID for progress tracking.
        """
        # Use TransNet V2 for scene detection
        scenes = []
        try:
            # 1. OPTIMIZATION: Check Cache (Avoids running TransNet per chunk)
            if self._cached_scenes and self._cached_scenes_path == str(path):
                scenes = self._cached_scenes
            else:
                # Run TransNet Logic (Once per file/trim)
                frame_scenes = self.transnet.predict_video(str(path))

                cap = cv2.VideoCapture(str(path))
                fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
                int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

                raw_scenes = []
                from core.processing.scene_detector import SceneInfo

                for start_frame, end_frame in frame_scenes:
                    # Use PTS for accurate timestamps instead of frame/fps
                    cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
                    start_msec = cap.get(cv2.CAP_PROP_POS_MSEC)
                    start_t = (
                        (start_msec / 1000.0)
                        if start_msec >= 0
                        else (start_frame / fps)
                    )

                    cap.set(cv2.CAP_PROP_POS_FRAMES, end_frame)
                    end_msec = cap.get(cv2.CAP_PROP_POS_MSEC)
                    end_t = (
                        (end_msec / 1000.0)
                        if end_msec >= 0
                        else (end_frame / fps)
                    )

                    if end_t - start_t >= 1.0:
                        raw_scenes.append(
                            SceneInfo(
                                start_time=start_t,
                                end_time=end_t,
                                start_frame=int(start_frame),
                                end_frame=int(end_frame),
                                mid_frame=int((start_frame + end_frame) / 2),
                                mid_time=(start_t + end_t) / 2,
                            )
                        )

                cap.release()

                if not raw_scenes:
                    logger.warning(
                        "TransNet returned no scenes, fallback to scenedetect"
                    )
                    raw_scenes = await detect_scenes(path)

                scenes = raw_scenes
                self._cached_scenes = scenes
                self._cached_scenes_path = str(path)

            # 2. CHUNK FILTERING (Critical for Memory/Efficiency)
            # Only process scenes that overlap with the current pipeline chunk context
            start_ctx = chunk_start if chunk_start is not None else 0.0
            end_ctx = chunk_end if chunk_end is not None else float("inf")

            # Filter scenes: [Scene Start < Chunk End] AND [Scene End > Chunk Start]
            filtered_scenes = [
                s
                for s in scenes
                if s.end_time > start_ctx and s.start_time < end_ctx
            ]

            if not filtered_scenes:
                logger.debug(
                    f"[Scenes] No scenes in chunk ({start_ctx}-{end_ctx})"
                )
                return

            scenes = filtered_scenes  # Proceed with filtered list

        except Exception as e:
            logger.error(f"TransNet detection failed: {e}")
            scenes = await detect_scenes(path)

        if not scenes:
            logger.info(f"No scene boundaries detected in {path.name}")
            return

        # Import aggregator
        from core.processing.scene_aggregator import aggregate_frames_to_scene

        vlm = get_vlm_client()
        prompt = (
            "Describe this scene in detail: actions, objects, colors, expressions, "
            "and atmosphere. Be specific about what people are doing and wearing."
        )

        # Get all audio segments for this video (for dialogue per scene)
        audio_segments = self._get_audio_segments_for_video(str(path))

        # Get audio events (sirens, alarms, etc.)
        audio_events = self._get_audio_events_for_video(str(path))

        # Get all frames for this video (for aggregation per scene)
        all_frames = self.db.get_frames_by_video(str(path))

        scenes_stored = 0
        for idx, scene in enumerate(scenes):
            if job_id and progress_tracker.is_cancelled(job_id):
                break

            # 1. Get VLM caption for representative frame
            visual_summary = ""
            frame_bytes = extract_scene_frame(path, scene.mid_time)
            if frame_bytes:
                try:
                    caption = vlm.generate_caption_from_bytes(
                        frame_bytes, prompt
                    )
                    if caption:
                        visual_summary = caption
                except Exception as e:
                    logger.warning(f"VLM caption failed for scene {idx}: {e}")

            # 2. Filter frames that belong to this scene
            scene_frames = [
                f
                for f in all_frames
                if scene.start_time <= f.get("timestamp", 0) <= scene.end_time
            ]

            # 3. Filter audio segments that overlap this scene
            scene_dialogue = [
                a
                for a in audio_segments
                if a.get("end", 0) > scene.start_time
                and a.get("start", 0) < scene.end_time
            ]

            # Filter audio events that overlap this scene
            scene_events = [
                e
                for e in audio_events
                if e.get("end", 0) > scene.start_time
                and e.get("start", 0) < scene.end_time
            ]

            # 4. Aggregate frame data into scene
            aggregated = aggregate_frames_to_scene(
                frames=scene_frames,
                start_time=scene.start_time,
                end_time=scene.end_time,
                dialogue_segments=scene_dialogue,
                audio_events=scene_events,
            )

            # Override with VLM summary if available
            if visual_summary:
                aggregated["visual_summary"] = visual_summary

            # 5. Build text for each vector
            # Visual: entities, clothing, people
            visual_parts = []
            for name in aggregated.get("person_names", []):
                visual_parts.append(name)
            for attr in aggregated.get("person_attributes", []):
                if attr.get("clothing_color") and attr.get("clothing_type"):
                    visual_parts.append(
                        f"{attr['clothing_color']} {attr['clothing_type']}"
                    )
                visual_parts.extend(attr.get("accessories", []))
            for entity in aggregated.get("entities", []):
                if isinstance(entity, dict):
                    visual_parts.append(entity.get("name", ""))
            visual_parts.extend(aggregated.get("visible_text", []))
            if visual_summary:
                visual_parts.append(visual_summary)

            # Explicitly add audio events for semantic search (e.g. "siren sound")
            audio_evts = aggregated.get("audio_events", [])
            if audio_evts:
                visual_parts.append(f"Sounds: {', '.join(audio_evts)}")

            visual_text = " ".join(filter(None, visual_parts))

            # Motion: actions
            motion_parts = aggregated.get("actions", [])
            if aggregated.get("action_sequence"):
                motion_parts.append(aggregated["action_sequence"])
            motion_text = " ".join(filter(None, motion_parts))

            # Dialogue: transcript
            dialogue_text = aggregated.get("dialogue_transcript", "")

            # 6. Deep Research Analysis removed per AGENTS.md clean architecture stance
            dr_meta = {}
            internvideo_features = None
            languagebind_features = None

            # 7. Generate CLIP/SigLIP visual features for true multimodal search
            visual_features = None
            if settings.enable_visual_embeddings and frame_bytes:
                try:
                    # Lazy load the encoder to save VRAM until needed
                    if self._visual_encoder is None:
                        from core.processing.visual_encoder import (
                            get_default_visual_encoder,
                        )

                        logger.info(
                            "Initializing Visual Encoder (CLIP/SigLIP) for ingestion..."
                        )
                        self._visual_encoder = get_default_visual_encoder()

                    # Convert frame bytes to numpy array
                    import io

                    from PIL import Image

                    img = Image.open(io.BytesIO(frame_bytes)).convert("RGB")
                    img_np = np.array(img)

                    # Encode the frame to get visual embeddings
                    visual_features_arr = (
                        await self._visual_encoder.encode_image(img_np)
                    )
                    if visual_features_arr is not None:
                        visual_features = (
                            visual_features_arr.tolist()
                            if hasattr(visual_features_arr, "tolist")
                            else list(visual_features_arr)
                        )
                        logger.debug(
                            f"Scene {idx}: Visual features generated (dim={len(visual_features)})"
                        )
                except Exception as e:
                    logger.warning(
                        f"Failed to generate visual features for scene {idx}: {e}"
                    )
                    visual_features = None

            # 8. Store scene with multi-vector
            try:
                # Save representative thumbnail
                thumb_path = None
                if frame_bytes:
                    thumb_dir = settings.cache_dir / "thumbnails" / "scenes"
                    thumb_dir.mkdir(parents=True, exist_ok=True)
                    safe_stem = hashlib.md5(path.stem.encode()).hexdigest()
                    thumb_name = f"{safe_stem}_{scene.start_time:.2f}.jpg"
                    thumb_file = thumb_dir / thumb_name
                    with open(thumb_file, "wb") as f:
                        f.write(frame_bytes)
                    thumb_path = f"/thumbnails/scenes/{thumb_name}"

                # Build payload
                payload = {
                    # Indexed for filtering
                    "face_cluster_ids": aggregated.get("face_cluster_ids", []),
                    "person_names": aggregated.get("person_names", []),
                    "clothing_colors": [
                        a.get("clothing_color", "")
                        for a in aggregated.get("person_attributes", [])
                        if a.get("clothing_color")
                    ],
                    "clothing_types": [
                        a.get("clothing_type", "")
                        for a in aggregated.get("person_attributes", [])
                        if a.get("clothing_type")
                    ],
                    "accessories": [
                        acc
                        for a in aggregated.get("person_attributes", [])
                        for acc in a.get("accessories", [])
                    ],
                    "actions": aggregated.get("actions", []),
                    "visible_text": aggregated.get("visible_text", []),
                    "entity_names": [
                        e.get("name", "")
                        for e in aggregated.get("entities", [])
                        if isinstance(e, dict)
                    ],
                    "audio_events": aggregated.get("audio_events", []),
                    # Deep Research Metadata
                    "shot_type": dr_meta.get("shot_type", ""),
                    "mood": dr_meta.get("mood", ""),
                    "aesthetic_score": dr_meta.get("aesthetic_score", 0.0),
                    # Non-indexed data
                    "visual_summary": visual_summary,
                    "action_sequence": aggregated.get("action_sequence", ""),
                    "location": aggregated.get("location", ""),
                    "cultural_context": aggregated.get("cultural_context", ""),
                    "dialogue_transcript": dialogue_text,
                    "frame_count": aggregated.get("frame_count", 0),
                    "thumbnail_path": thumb_path,
                }

                # Enhance visual text with Deep Research insights
                # Only add shot_type and mood — these are semantic terms that help
                # text embedding search (e.g. "close-up", "tense").
                # aesthetic_score is NOT added here because numeric strings
                # ("aesthetic_score: 0.75") are noise in text embeddings.
                # The score is already stored as a filterable numeric field.
                if dr_meta:
                    dr_parts = []
                    if dr_meta.get("shot_type"):
                        dr_parts.append(dr_meta["shot_type"])
                    if dr_meta.get("mood"):
                        dr_parts.append(dr_meta["mood"])
                    if dr_parts:
                        visual_text += " " + " ".join(dr_parts)

                await self.db.store_scene(
                    media_path=str(path),
                    start_time=scene.start_time,
                    end_time=scene.end_time,
                    visual_text=visual_text,
                    motion_text=motion_text,
                    dialogue_text=dialogue_text,
                    visual_features=visual_features,  # CLIP/SigLIP embedding
                    internvideo_features=internvideo_features,
                    languagebind_features=languagebind_features,
                    payload=payload,
                )
                scenes_stored += 1

                # === NEO4J GRAPH INGESTION (Prod Ready Abuse) ===
                try:
                    # Initialize Video Node (idempotent)
                    if scenes_stored == 1:
                        from core.domain.schemas import (
                            MediaFile,
                            MediaMetadata,
                # Graph updates handled in SQL system of record
                pass

            except Exception as e:
                logger.warning(f"Failed to store scene {idx}: {e}")

        logger.info(
            f"Stored {scenes_stored}/{len(scenes)} scenes for {path.name}"
        )
