"""Scene repository - scene, scenelet, and masklet CRUD operations.

Extracted from VectorDB to reduce God class size.
VectorDB inherits from SceneRepository to compose these methods.
"""

from __future__ import annotations

import uuid
from typing import TYPE_CHECKING, Any

import numpy as np
from qdrant_client.http import models

from config import settings
from core.storage.constants import (
    MASKLETS,
    MASKLETS_COLLECTION,
    MEDIA_COLLECTION,
    MEDIA_FRAMES,
    MEDIA_VECTOR_SIZE,
    SCENELETS_COLLECTION,
    SCENES_COLLECTION,
    SCENE_EMBEDDINGS,
    SCENE_MULTI_VECTOR_DIM,
    TEXT_DIM,
)
from core.storage.qdrant_utils import paginated_scroll, retry_on_connection_error
from core.utils.logger import log

if TYPE_CHECKING:
    from qdrant_client import QdrantClient


class SceneRepository:
    """Scene detection, captioning, masklet tracking, and scenelet operations."""

    client: QdrantClient

    async def search_scenelets(
        self,
        query: str,
        limit: int = 10,
        video_path: str | None = None,
        gap_threshold: float = 2.0,
        padding: float = 3.0,
    ) -> list[dict]:
        """Search for dynamic video segments based on dense frame retrieval.

        Performs semantic search on frames, then clusters temporally adjacent
        matches to form coherent 'scenelets'. Applies padding to capture context.

        Args:
            query: Natural language search query.
            limit: Number of final scenelets to return.
            video_path: Optional filter for specific video.
            gap_threshold: Max seconds between frames to merge into one cluster.
            padding: Seconds to add before start and after end.

        Returns:
            List of dicts with 'start_time', 'end_time', 'score', etc.
        """
        # 1. Retrieve raw frame candidates
        # We fetch more than limit because clusters will reduce count
        raw_limit = limit * 5

        # Reuse existing frame search logic but get raw points
        # FIX: Use Visual Encoder for Scenelets (based on Frames)
        try:
            query_vector = await self.visual_encoder.encode_text(query)
            if not query_vector:
                log("Visual encoder returned empty for scenelet query, skipping vector search", level="WARNING")
                return []
        except Exception:
            log("Visual encoder unavailable for scenelet search, returning empty", level="WARNING")
            return []

        filters = []
        if video_path:
            filters.append(
                models.FieldCondition(
                    key="video_path",
                    match=models.MatchValue(value=video_path),
                )
            )

        try:
            # FIX: Use query_points instead of search (deprecated/wrong method name for Client)
            resp = self.client.query_points(
                collection_name=self.MEDIA_COLLECTION,
                query=query_vector,
                limit=raw_limit,
                query_filter=models.Filter(must=filters) if filters else None,
                with_payload=True,
            )
            hits = resp.points
        except Exception as e:
            log(f"[Scenelet] Frame search failed: {e}", level="ERROR")
            return []

        if not hits:
            return []

        # 2. Group by Video
        # (Though usually we search one video or global, let's handle mixed)
        hits_by_video = {}
        for hit in hits:
            payload = hit.payload or {}
            v_path = payload.get("video_path")
            if not v_path:
                continue

            if v_path not in hits_by_video:
                hits_by_video[v_path] = []

            # Extract timestamp
            ts = payload.get("timestamp")
            if ts is None:
                # Try start_time alias
                ts = payload.get("start_time", 0.0)

            hits_by_video[v_path].append(
                {
                    "score": hit.score,
                    "timestamp": float(ts),
                    "text": payload.get("description", "")
                    or payload.get("ocr_text", "")
                    or "",
                    "payload": payload,
                }
            )

        final_scenelets = []

        # 3. Clustering & Padding per Video
        for v_path, candidates in hits_by_video.items():
            # Sort by timestamp
            candidates.sort(key=lambda x: x["timestamp"])

            clusters = []
            if not candidates:
                continue

            # Current cluster state
            current_cluster = [candidates[0]]

            for i in range(1, len(candidates)):
                curr = candidates[i]
                prev = candidates[i - 1]

                # Check time gap
                if (curr["timestamp"] - prev["timestamp"]) <= gap_threshold:
                    current_cluster.append(curr)
                else:
                    # Finalize current cluster
                    clusters.append(current_cluster)
                    current_cluster = [curr]

            # Append last cluster
            if current_cluster:
                clusters.append(current_cluster)

            # Process Clusters into Scenelets
            for cl in clusters:
                # Core bounds
                start_ts = cl[0]["timestamp"]
                end_ts = cl[-1]["timestamp"]

                # Scores
                max_score = max(c["score"] for c in cl)
                avg_score = sum(c["score"] for c in cl) / len(cl)

                # Descriptions (Best score's desc)
                best_frame = max(cl, key=lambda x: x["score"])
                description = best_frame["text"]

                # Pad
                final_start = max(0.0, start_ts - padding)
                # Note: We need video duration to clamp end efficiently.
                # For now, we rely on UI to handle over-bounds or just let it be.
                # Or check if metadata exists. A simple approach is just setting it.
                final_end = end_ts + padding

                final_scenelets.append(
                    {
                        "video_path": v_path,
                        "start_time": final_start,
                        "end_time": final_end,
                        "core_start": start_ts,
                        "core_end": end_ts,
                        "score": max_score,  # Use max for retrieval ranking
                        "text": description,
                        "frame_count": len(cl),
                        "best_frame_timestamp": best_frame["timestamp"],
                    }
                )

        # 4. Sort and Limit
        final_scenelets.sort(key=lambda x: x["score"], reverse=True)
        return final_scenelets[:limit]

    def insert_masklet(
        self,
        video_path: str,
        concept: str,
        start_time: float,
        end_time: float,
        confidence: float = 1.0,
        payload: dict[str, Any] | None = None,
        embedding: list[float] | None = None,
    ) -> None:
        """Inserts a masklet (video segment tracking a specific concept).

        Args:
            video_path: Path to the source video file.
            concept: The concept name/label being tracked.
            start_time: Start timestamp of the masklet in seconds.
            end_time: End timestamp of the masklet in seconds.
            confidence: Confidence score of the tracking.
            payload: Optional additional metadata for the masklet.
            embedding: Visual embedding vector (1024d/1152d) for SigLIP search.
        """
        unique_str = f"{video_path}_{concept}_{start_time}_{end_time}"
        point_id = str(uuid.uuid5(uuid.NAMESPACE_DNS, unique_str))

        final_payload = {
            "video_path": video_path,
            "concept": concept,
            "start_time": start_time,
            "end_time": end_time,
            "confidence": confidence,
            "type": "masklet",
        }
        if payload:
            final_payload.update(payload)

        # Sanitize for Qdrant compatibility
        final_payload = sanitize_numpy_types(final_payload)

        # Use provided embedding or dummy fallback (not ideal for search)
        vector = embedding if embedding else [0.0] * self.MEDIA_VECTOR_SIZE

        try:
            self.client.upsert(
                collection_name=self.MASKLETS_COLLECTION,
                points=[
                    models.PointStruct(
                        id=point_id,
                        vector=vector,
                        payload=final_payload,
                    )
                ],
                wait=False,
            )
        except Exception as e:
            log(f"Failed to insert masklet: {e}", level="ERROR")

    async def search_masklets(
        self,
        query: str | None = None,
        query_vector: list[float] | None = None,
        concept: str | None = None,
        limit: int = 10,
        video_path: str | None = None,
        score_threshold: float | None = None,
    ) -> list[dict[str, Any]]:
        """Search masklets (tracked concepts) by text, vector, or exact concept.

        Args:
            query: Natural language query (e.g., "red car").
            query_vector: Pre-computed visual embedding.
            concept: Exact concept name filter.
            limit: Result limit.
            video_path: Optional filter by source video.
            score_threshold: Minimum similarity score.
        """
        # 1. Resolve Query Vector
        if query and not query_vector:
            query_vector = await self.encode_text(query)

        # 2. Build Filters
        conditions = []
        if concept:
            conditions.append(
                models.FieldCondition(
                    key="concept", match=models.MatchValue(value=concept)
                )
            )
        if video_path:
            conditions.append(media_path_filter(video_path))

        query_filter = build_filter(conditions) if conditions else None

        # 3. Execute Search or Scroll
        try:
            if query_vector:
                results = self.client.search(
                    collection_name=self.MASKLETS_COLLECTION,
                    query_vector=query_vector,
                    query_filter=query_filter,
                    limit=limit,
                    score_threshold=score_threshold,
                    with_payload=True,
                )
            else:
                # Fallback to scroll if no vector provided (Exact Concept lookup)
                resp, _ = self.client.scroll(
                    collection_name=self.MASKLETS_COLLECTION,
                    scroll_filter=query_filter,
                    limit=limit,
                    with_payload=True,
                )
                # Mock result objects for consistency
                results = [
                    type("Point", (), {"id": p.id, "score": 1.0, "payload": p.payload})
                    for p in resp
                ]

            return [
                {
                    "id": str(r.id),
                    "score": r.score,
                    "video_path": r.payload.get("video_path") or r.payload.get("media_path", ""),
                    "concept": r.payload.get("concept", ""),
                    "start_time": r.payload.get("start_time") or r.payload.get("start", 0),
                    "end_time": r.payload.get("end_time") or r.payload.get("end", 0),
                    "confidence": r.payload.get("confidence", 1.0),
                    **r.payload,
                }
                for r in results
                if r.payload
            ]
        except Exception as e:
            log(f"search_masklets failed: {e}", level="ERROR")
            return []

    async def update_masklet_concept(
        self,
        old_concept: str,
        new_concept: str,
    ) -> int:
        """Updates the concept name for all matching masklets (Renaming).

        Used when an identity is renamed to ensure SAM 3 tracks are updated.
        """
        try:
            # 1. Find all masklets with old_concept (Scroll)
            masklets = await self.search_masklets(concept=old_concept, limit=1000)
            if not masklets:
                return 0

            count = 0
            for m in masklets:
                point_id = m["id"]
                # Qdrant set_payload allows partial updates
                self.client.set_payload(
                    collection_name=self.MASKLETS_COLLECTION,
                    payload={"concept": new_concept},
                    points=[point_id],
                )
                count += 1

            log(f"Renamed {count} masklets from '{old_concept}' to '{new_concept}'")
            return count

        except Exception as e:
            log(f"Failed to update masklet concepts: {e}", level="ERROR")
            return 0

    def update_masklet(
        self,
        masklet_id: str,
        updates: dict[str, Any],
    ) -> bool:
        """Updates an existing masklet payload.

        Args:
            masklet_id: The ID of the masklet point.
            updates: Dictionary of fields to update in the payload.

        Returns:
            True if successful.
        """
        try:
            # We use set_payload to update specific fields without rewriting the whole point
            self.client.set_payload(
                collection_name=self.MASKLETS_COLLECTION,
                payload=updates,
                points=[masklet_id],
            )
            return True
        except Exception as e:
            log(f"Failed to update masklet {masklet_id}: {e}", level="ERROR")
            return False

    def get_masklets(
        self,
        video_path: str,
        start_time: float | None = None,
        end_time: float | None = None,
    ) -> list[dict[str, Any]]:
        """Retrieves masklets for a specific video and optional time range.

        Args:
            video_path: Path to the source video.
            start_time: Optional start search bound.
            end_time: Optional end search bound.

        Returns:
            List of masklet payloads.
        """
        must_filters = [
            models.FieldCondition(
                key="video_path", match=models.MatchValue(value=video_path)
            ),
            models.FieldCondition(
                key="type", match=models.MatchValue(value="masklet")
            ),
        ]

        if start_time is not None:
            must_filters.append(
                models.FieldCondition(
                    key="start_time", range=models.Range(gte=start_time)
                )
            )
        if end_time is not None:
            must_filters.append(
                models.FieldCondition(
                    key="end_time", range=models.Range(lte=end_time)
                )
            )

        try:
            results = self.client.scroll(
                collection_name=self.MASKLETS_COLLECTION,
                scroll_filter=models.Filter(
                    must=cast(list[models.Condition], must_filters)
                ),
                limit=1000,
            )
            # Ensure payloads are not None before returning
            valid_payloads: list[dict[str, Any]] = [
                p.payload for p in results[0] if p.payload is not None
            ]
            return valid_payloads
        except Exception as e:
            log(f"Failed to fetch masklets: {e}", level="ERROR")
            return []

    async def store_scene(
        self,
        media_path: str,
        start_time: float,
        end_time: float,
        visual_text: str = "",
        motion_text: str = "",
        dialogue_text: str = "",
        visual_features: list[float]
        | None = None,  # Actual visual embedding (CLIP/SigLIP)
        internvideo_features: list[float]
        | None = None,  # InternVideo action embedding
        languagebind_features: list[float]
        | None = None,  # LanguageBind multimodal embedding
        payload: dict[str, Any] | None = None,
    ) -> str:
        """Store a scene with multi-vector embeddings (visual, motion, dialogue).

        This is the production-grade approach used by Twelve Labs Marengo.
        Each scene gets 6 vectors:
        - visual: Text embedding of visual description
        - motion: Text embedding of motion/action description
        - dialogue: Text embedding of dialogue/transcript
        - visual_features: ACTUAL visual embedding from CLIP/SigLIP (for image-as-query)
        - internvideo: InternVideo2 action embedding (for action queries)
        - languagebind: LanguageBind multimodal embedding (text-aligned video)

        Args:
            media_path: Path to the source video.
            start_time: Scene start timestamp in seconds.
            end_time: Scene end timestamp in seconds.
            visual_text: Text describing visual content (entities, clothing, people).
            motion_text: Text describing actions and movement.
            dialogue_text: Transcript/dialogue for this scene.
            visual_features: Optional actual visual embedding from visual encoder.
            internvideo_features: Optional InternVideo action embedding.
            languagebind_features: Optional LanguageBind multimodal embedding.
            payload: Additional structured data (SceneData.to_payload()).

        Returns:
            The generated scene ID.
        """
        import numpy as np
        _NEAR_ZERO = float(np.finfo(np.float32).tiny)

        def _safe_fill(dim: int) -> list[float]:
            """Near-zero vector that avoids NaN cosine similarity."""
            return [_NEAR_ZERO] * dim

        def _adapt_features(vec: list[float], expected_dim: int, name: str) -> list[float]:
            """Pad or truncate feature vectors on dim mismatch instead of discarding."""
            if len(vec) == expected_dim:
                return vec
            log(
                f"{name} dim adapted: got {len(vec)}, expected {expected_dim}",
                level="WARNING",
            )
            if len(vec) < expected_dim:
                return vec + [_NEAR_ZERO] * (expected_dim - len(vec))
            return vec[:expected_dim]

        # Generate text embeddings ONLY for non-empty text
        if visual_text:
            visual_vec = (await self.encode_texts(visual_text))[0]
        else:
            visual_vec = _safe_fill(self.TEXT_DIM)

        if motion_text:
            motion_vec = (await self.encode_texts(motion_text))[0]
        else:
            motion_vec = _safe_fill(self.TEXT_DIM)

        if dialogue_text:
            dialogue_vec = (await self.encode_texts(dialogue_text))[0]
        else:
            dialogue_vec = _safe_fill(self.TEXT_DIM)

        # Visual features (actual visual embedding from CLIP/SigLIP)
        visual_features_dim = getattr(settings, "visual_features_dim", 768)
        video_embedding_dim = getattr(settings, "video_embedding_dim", 1024)

        if visual_features is None:
            visual_features = _safe_fill(visual_features_dim)
        else:
            visual_features = _adapt_features(visual_features, visual_features_dim, "Visual features")

        if internvideo_features is None:
            internvideo_features = _safe_fill(video_embedding_dim)
        else:
            internvideo_features = _adapt_features(internvideo_features, video_embedding_dim, "InternVideo features")

        if languagebind_features is None:
            languagebind_features = _safe_fill(video_embedding_dim)
        else:
            languagebind_features = _adapt_features(languagebind_features, video_embedding_dim, "LanguageBind features")

        # Generate unique scene ID
        scene_key = f"{media_path}_{start_time:.3f}_{end_time:.3f}"
        scene_id = str(uuid.uuid5(uuid.NAMESPACE_DNS, scene_key))

        # Build full payload with modality presence flags for search filtering
        _has_real_values = lambda v: any(abs(x) > _NEAR_ZERO for x in v[:10])
        full_payload = {
            "media_path": media_path,
            "start_time": start_time,
            "end_time": end_time,
            "duration": end_time - start_time,
            "visual_text": visual_text,
            "motion_text": motion_text,
            "dialogue_text": dialogue_text,
            # Modality presence flags — search can skip empty modalities
            "has_visual_text": bool(visual_text),
            "has_motion_text": bool(motion_text),
            "has_dialogue_text": bool(dialogue_text),
            "has_visual_features": _has_real_values(visual_features),
            "has_internvideo": _has_real_values(internvideo_features),
            "has_languagebind": _has_real_values(languagebind_features),
        }
        if payload:
            full_payload.update(payload)

        # Prepare the vector dictionary — all vectors guaranteed correct dim
        vector_dict = {
            "visual": visual_vec,
            "motion": motion_vec,
            "dialogue": dialogue_vec,
            "visual_features": visual_features,
            "internvideo": internvideo_features,
            "languagebind": languagebind_features,
        }

        self.client.upsert(
            collection_name=self.SCENES_COLLECTION,
            points=[
                models.PointStruct(
                    id=scene_id,
                    vector=vector_dict,
                    payload=full_payload,
                )
            ],
        )

        log(
            f"Stored scene {start_time:.1f}-{end_time:.1f}s for {Path(media_path).name}"
        )
        return scene_id

    async def store_scenelet(
        self,
        *,
        media_path: str,
        start_time: float,
        end_time: float,
        content_text: str,
        payload: dict[str, Any] | None = None,
    ) -> str:
        """Store a sliding window scenelet.

        Args:
            media_path: Source video path.
            start_time: Start timestamp.
            end_time: End timestamp.
            content_text: Fused text (Visual + Audio).
            payload: Additional metadata.
        """
        vector = (await self.encode_texts(content_text or "scenelet"))[0]

        scenelet_id = str(
            uuid.uuid5(uuid.NAMESPACE_DNS, f"{media_path}_sl_{start_time:.3f}")
        )

        full_payload = {
            "media_path": media_path,
            "start_time": start_time,
            "end_time": end_time,
            "text": content_text,
        }
        if payload:
            full_payload.update(payload)

        self.client.upsert(
            collection_name=self.SCENELETS_COLLECTION,
            points=[
                models.PointStruct(
                    id=scenelet_id,
                    vector={"content": vector},
                    payload=full_payload,
                )
            ],
        )
        return scenelet_id

    async def search_scenes(
        self,
        query: str | list[float],
        *,
        limit: int = 20,
        score_threshold: float | None = None,
        # Identity filters - FIX: Accept list of names instead of single name
        person_names: list[str] | None = None,
        person_name: str | None = None,  # DEPRECATED: kept for backwards compat
        face_cluster_ids: list[int] | None = None,
        # Clothing/appearance filters — now accept LISTS (Fix #6, #11)
        clothing_colors: list[str] | None = None,
        clothing_types: list[str] | None = None,
        clothing_color: str | None = None,   # DEPRECATED: kept for backwards compat
        clothing_type: str | None = None,     # DEPRECATED: kept for backwards compat
        accessories: list[str] | None = None,
        # Content filters
        location: str | None = None,
        visible_text: list[str] | None = None,
        # Action filters
        action_keywords: list[str] | None = None,
        # Deep Research Filters
        mood: str | None = None,
        shot_type: str | None = None,
        aesthetic_score: float | None = None,
        # Exclusion filters (Fix #5)
        exclusions: list[dict[str, Any]] | None = None,
        # Time filters
        video_path: str | None = None,
        video_paths: list[str] | None = None,
        # Search mode
        search_mode: str = "hybrid",  # "visual", "motion", "dialogue", "hybrid"
    ) -> list[dict[str, Any]]:
        """Search scenes with comprehensive filtering for complex queries.

        Supports queries like:
        - "Prakash wearing blue shirt bowling at Brunswick hitting a strike"

        Args:
            query: Natural language search query.
            limit: Maximum results.
            score_threshold: Minimum similarity score.
            person_names: Filter by person names (list). Matches ANY name in list.
            person_name: DEPRECATED. Use person_names instead.
            face_cluster_ids: Filter by face clusters.
            clothing_color: Filter by clothing color (e.g., "blue").
            clothing_type: Filter by clothing type (e.g., "shirt").
            accessories: Filter by accessories (e.g., ["spectacles"]).
            location: Filter by location (e.g., "Brunswick").
            visible_text: Filter by visible text/brands.
            action_keywords: Filter by actions.
            video_path: Filter by specific video.
            search_mode: Which vector(s) to search.

        Returns:
            List of matching scenes with timestamps and metadata.
        """
        # Build query vector based on mode
        if isinstance(query, str):
            query_vec = (await self.encode_texts(query, is_query=True))[0]
        else:
            query_vec = query

        # Build filter conditions
        conditions: list[models.Condition] = []

        # Video filter
        if video_path:
            conditions.append(
                models.FieldCondition(
                    key="media_path",
                    match=models.MatchValue(value=video_path),
                )
            )

        # Hierarchical Filter (Multiple Videos)
        if video_paths:
            conditions.append(
                models.FieldCondition(
                    key="media_path",
                    match=models.MatchAny(any=video_paths),
                )
            )

        # Identity filter - FIX: Support multiple person names
        # Merge person_names list with deprecated person_name for backwards compat
        all_person_names = list(person_names) if person_names else []
        if person_name and person_name not in all_person_names:
            all_person_names.append(person_name)

        if all_person_names:
            conditions.append(
                models.FieldCondition(
                    key="person_names",
                    match=models.MatchAny(
                        any=all_person_names
                    ),  # Match ANY name
                )
            )

        if face_cluster_ids:
            conditions.append(
                models.FieldCondition(
                    key="face_cluster_ids",
                    match=models.MatchAny(any=face_cluster_ids),
                )
            )

        # Clothing/appearance filters (Fix #6, #11: accept lists)
        # Merge deprecated single-value params into lists
        all_clothing_colors = list(clothing_colors) if clothing_colors else []
        if clothing_color and clothing_color.lower() not in all_clothing_colors:
            all_clothing_colors.append(clothing_color.lower())
        all_clothing_types = list(clothing_types) if clothing_types else []
        if clothing_type and clothing_type.lower() not in all_clothing_types:
            all_clothing_types.append(clothing_type.lower())

        # For multi-clothing: add one condition per item (AND logic)
        # Searches clothing_descriptions (full VLM text) via substring matching
        for color in all_clothing_colors:
            conditions.append(
                models.FieldCondition(
                    key="clothing_descriptions",
                    match=models.MatchText(text=color),
                )
            )
        for ctype in all_clothing_types:
            conditions.append(
                models.FieldCondition(
                    key="clothing_types",
                    match=models.MatchText(text=ctype),
                )
            )

        if accessories:
            conditions.append(
                models.FieldCondition(
                    key="accessories",
                    match=models.MatchAny(any=accessories),
                )
            )

        # Location filter
        if location:
            conditions.append(
                models.FieldCondition(
                    key="location",
                    match=models.MatchText(text=location),
                )
            )

        # Visible text/brand filter — use MatchText for substring matching
        if visible_text:
            for vt in visible_text:
                conditions.append(
                    models.FieldCondition(
                        key="visible_text",
                        match=models.MatchText(text=vt),
                    )
                )

        # Action filter — use MatchText for fuzzy substring matching
        # "bowling" will match stored action "person is bowling at the alley"
        if action_keywords:
            for action in action_keywords:
                conditions.append(
                    models.FieldCondition(
                        key="actions",
                        match=models.MatchText(text=action),
                    )
                )

        # Deep Research Filters (Cinematography) — MatchText for fuzzy matching
        if mood:
            conditions.append(
                models.FieldCondition(
                    key="mood", match=models.MatchText(text=mood)
                )
            )

        if shot_type:
            conditions.append(
                models.FieldCondition(
                    key="shot_type", match=models.MatchText(text=shot_type)
                )
            )

        if aesthetic_score is not None:
            conditions.append(
                models.FieldCondition(
                    key="aesthetic_score",
                    range=models.Range(gte=aesthetic_score),
                )
            )

        # Build exclusion (must_not) conditions (Fix #5)
        must_not_conditions: list[models.Condition] = []
        EXCLUSION_FIELD_MAP = {
            "action": "actions",
            "person": "person_names",
            "location": "location",
            "mood": "mood",
            "clothing": "clothing_types",
            "audio": "audio_events",
            "text": "visible_text",
            "accessory": "accessories",
        }
        if exclusions:
            for exc in exclusions:
                exc_type = exc.get("type", "").lower()
                exc_value = exc.get("value", "")
                if not exc_value:
                    continue
                field_key = EXCLUSION_FIELD_MAP.get(exc_type)
                if field_key:
                    must_not_conditions.append(
                        models.FieldCondition(
                            key=field_key,
                            match=models.MatchText(text=exc_value),
                        )
                    )

        # Build final filter
        query_filter = models.Filter(
            must=conditions if conditions else None,
            must_not=must_not_conditions if must_not_conditions else None,
        ) if (conditions or must_not_conditions) else None

        # Execute search based on mode
        results = []

        if search_mode == "hybrid":
            # Search all enabled vectors and combine results
            # Core text vectors (always searched)
            target_vectors = ["visual", "motion", "dialogue"]

            # Video understanding vectors (if enabled and stored)
            if getattr(settings, "enable_video_embeddings", True):
                target_vectors.extend(["internvideo", "languagebind"])

            # Visual features (CLIP/SigLIP) if enabled
            if getattr(settings, "enable_visual_features", True):
                target_vectors.append("visual_features")

            for vector_name in target_vectors:
                try:
                    # For video/visual vectors, add filter to only search scenes with those embeddings
                    search_filter = query_filter
                    if vector_name in [
                        "internvideo",
                        "languagebind",
                        "visual_features",
                    ]:
                        has_key = f"has_{vector_name}"
                        embedding_filter = models.FieldCondition(
                            key=has_key,
                            match=models.MatchValue(value=True),
                        )
                        if search_filter:
                            # Combine with existing filter
                            search_filter = models.Filter(
                                must=[*search_filter.must, embedding_filter]
                                if search_filter.must
                                else [embedding_filter]
                            )
                        else:
                            search_filter = models.Filter(
                                must=[embedding_filter]
                            )

                    resp = self.client.query_points(
                        collection_name=self.SCENES_COLLECTION,
                        query=query_vec,
                        using=vector_name,
                        limit=limit,
                        score_threshold=score_threshold,
                        query_filter=search_filter,
                    )
                    for hit in resp.points:
                        results.append(
                            {
                                "score": hit.score,
                                "id": str(hit.id),
                                "vector_type": vector_name,
                                **(hit.payload or {}),
                            }
                        )
                except Exception as e:
                    log(f"Scene search ({vector_name}) error: {e}")

            # Dedupe and sort by highest score
            seen_ids = set()
            unique_results = []
            for r in sorted(results, key=lambda x: x["score"], reverse=True):
                if r["id"] not in seen_ids:
                    seen_ids.add(r["id"])
                    unique_results.append(r)
            results = unique_results[:limit]

        else:
            # Single vector search
            try:
                resp = self.client.query_points(
                    collection_name=self.SCENES_COLLECTION,
                    query=query_vec,
                    using=search_mode,
                    limit=limit,
                    score_threshold=score_threshold,
                    query_filter=query_filter,
                )
                for hit in resp.points:
                    results.append(
                        {
                            "score": hit.score,
                            "id": str(hit.id),
                            "vector_type": search_mode,
                            **(hit.payload or {}),
                        }
                    )
            except Exception as e:
                log(f"Scene search ({search_mode}) error: {e}")

        # HITL Name Confidence Boosting
        # Boost scores for results containing HITL-named identities that match the query
        # FIX: Use all_person_names list instead of single person_name
        if all_person_names:
            query_names_lower = [n.lower() for n in all_person_names]
            for result in results:
                # Check if result has face_names or person_names that match
                face_names = result.get("face_names", []) or result.get(
                    "person_names", []
                )
                if face_names:
                    for name in face_names:
                        if name and any(
                            qn in name.lower() for qn in query_names_lower
                        ):
                            # 50% boost for exact HITL name match
                            result["score"] = result.get("score", 0) * 1.5
                            result["hitl_boost"] = True
                            result["matched_person"] = (
                                name  # Track which person matched
                            )
                            break

        # Re-sort after boosting
        results.sort(key=lambda x: x.get("score", 0), reverse=True)

        return results

    async def search_scenes_by_image(
        self,
        image: np.ndarray | bytes | Path,
        limit: int = 10,
        video_path: str | None = None,
        score_threshold: float | None = None,
    ) -> list[dict]:
        """Search scenes using an image as query (true visual search).

        This enables queries like "find scenes that look like this image"
        by using actual visual embeddings (CLIP/SigLIP) instead of text.

        Args:
            image: Query image (numpy array, bytes, or path).
            limit: Maximum results.
            video_path: Optional filter by video.
            score_threshold: Minimum similarity.

        Returns:
            List of matching scenes with scores.
        """
        from core.processing.visual_encoder import get_default_visual_encoder

        # Encode query image
        encoder = get_default_visual_encoder()
        query_vector = await encoder.encode_image(image)

        # Build filter
        filters = []
        if video_path:
            filters.append(
                models.FieldCondition(
                    key="media_path",
                    match=models.MatchValue(value=video_path),
                )
            )
        # Only search scenes with actual visual features
        filters.append(
            models.FieldCondition(
                key="has_visual_features",
                match=models.MatchValue(value=True),
            )
        )

        try:
            results = self.client.search(
                collection_name=self.SCENES_COLLECTION,
                query_vector=models.NamedVector(
                    name="visual_features",
                    vector=query_vector,
                ),
                limit=limit,
                score_threshold=score_threshold,
                query_filter=models.Filter(must=filters) if filters else None,
            )

            return [
                {
                    "id": str(r.id),
                    "score": r.score,
                    "media_path": r.payload.get("media_path", ""),
                    "start_time": r.payload.get("start_time", 0),
                    "end_time": r.payload.get("end_time", 0),
                    "visual_text": r.payload.get("visual_text", ""),
                    "search_mode": "image",
                    **r.payload,
                }
                for r in results
                if r.payload
            ]
        except Exception as e:
            log(f"Image search failed: {e}")
            return []

    async def search_scenes_by_action(
        self,
        query: str,
        limit: int = 10,
        video_path: str | None = None,
        score_threshold: float | None = None,
        search_mode: str = "hybrid",  # "internvideo", "languagebind", "hybrid"
    ) -> list[dict]:
        """Search scenes by action/motion using video embeddings.

        This uses InternVideo (action recognition) and LanguageBind (multimodal)
        embeddings for queries like "person kicking ball" or "car driving fast".

        Args:
            query: Action/motion query text.
            limit: Maximum results.
            video_path: Optional filter by video.
            score_threshold: Minimum similarity.
            search_mode: Which embedding to use (internvideo/languagebind/hybrid).

        Returns:
            List of matching scenes with scores.
        """
        from core.processing.video_understanding import LanguageBindEncoder

        # Build filter
        filters = []
        if video_path:
            filters.append(
                models.FieldCondition(
                    key="media_path",
                    match=models.MatchValue(value=video_path),
                )
            )

        results = []

        # Get text embedding for query (LanguageBind is text-aligned)
        try:
            encoder = LanguageBindEncoder()
            query_embedding = await encoder.encode_text(query)

            if query_embedding is None:
                # Fallback to standard text encoding
                query_embedding = (await self.encode_texts(query))[0]

            query_embedding = (
                list(query_embedding) if query_embedding is not None else None
            )
        except Exception as e:
            log(f"Failed to encode action query: {e}")
            query_embedding = (await self.encode_texts(query))[0]

        if query_embedding is None:
            return []

        # Search based on mode
        if search_mode in ("languagebind", "hybrid"):
            try:
                # Add filter for scenes with LanguageBind embeddings
                lb_filters = filters.copy()
                lb_filters.append(
                    models.FieldCondition(
                        key="has_languagebind",
                        match=models.MatchValue(value=True),
                    )
                )

                lb_results = self.client.search(
                    collection_name=self.SCENES_COLLECTION,
                    query_vector=models.NamedVector(
                        name="languagebind",
                        vector=query_embedding,
                    ),
                    limit=limit,
                    score_threshold=score_threshold,
                    query_filter=models.Filter(must=lb_filters)
                    if lb_filters
                    else None,
                )

                for r in lb_results:
                    if r.payload:
                        results.append(
                            {
                                "id": str(r.id),
                                "score": r.score,
                                "media_path": r.payload.get("media_path", ""),
                                "start_time": r.payload.get("start_time", 0),
                                "end_time": r.payload.get("end_time", 0),
                                "motion_text": r.payload.get("motion_text", ""),
                                "search_mode": "languagebind",
                                "source": "languagebind",
                                **r.payload,
                            }
                        )
            except Exception as e:
                log(f"LanguageBind search failed: {e}")

        if search_mode in ("internvideo", "hybrid"):
            try:
                # Add filter for scenes with InternVideo embeddings
                iv_filters = filters.copy()
                iv_filters.append(
                    models.FieldCondition(
                        key="has_internvideo",
                        match=models.MatchValue(value=True),
                    )
                )

                iv_results = self.client.search(
                    collection_name=self.SCENES_COLLECTION,
                    query_vector=models.NamedVector(
                        name="internvideo",
                        vector=query_embedding,
                    ),
                    limit=limit,
                    score_threshold=score_threshold,
                    query_filter=models.Filter(must=iv_filters)
                    if iv_filters
                    else None,
                )

                for r in iv_results:
                    if r.payload:
                        results.append(
                            {
                                "id": str(r.id),
                                "score": r.score,
                                "media_path": r.payload.get("media_path", ""),
                                "start_time": r.payload.get("start_time", 0),
                                "end_time": r.payload.get("end_time", 0),
                                "motion_text": r.payload.get("motion_text", ""),
                                "search_mode": "internvideo",
                                "source": "internvideo",
                                **r.payload,
                            }
                        )
            except Exception as e:
                log(f"InternVideo search failed: {e}")

        # RRF fusion if hybrid
        if search_mode == "hybrid" and results:
            # Deduplicate by ID, keeping highest score
            seen = {}
            for r in results:
                rid = r["id"]
                if rid not in seen or r["score"] > seen[rid]["score"]:
                    seen[rid] = r
            results = sorted(
                seen.values(), key=lambda x: x["score"], reverse=True
            )

        return results[:limit]

    def get_scene_by_id(self, scene_id: str) -> dict[str, Any] | None:
        """Get a scene by its ID.

        Args:
            scene_id: The scene ID.

        Returns:
            Scene data or None if not found.
        """
        try:
            points = self.client.retrieve(
                collection_name=self.SCENES_COLLECTION,
                ids=[scene_id],
                with_payload=True,
            )
            if points:
                return {"id": scene_id, **(points[0].payload or {})}
            return None
        except Exception:
            return None

    def get_scenes_for_video(
        self,
        video_path: str,
        limit: int = 100,
    ) -> list[dict[str, Any]]:
        """Get all scenes for a video, ordered by start time.

        Args:
            video_path: Path to the video.
            limit: Maximum results.

        Returns:
            List of scenes ordered by start_time.
        """
        try:
            resp = self.client.scroll(
                collection_name=self.SCENES_COLLECTION,
                scroll_filter=models.Filter(
                    must=[
                        models.FieldCondition(
                            key="media_path",
                            match=models.MatchValue(value=video_path),
                        )
                    ]
                ),
                limit=limit,
                with_payload=True,
            )
            scenes = []
            for point in resp[0]:
                scenes.append(
                    {
                        "id": str(point.id),
                        **(point.payload or {}),
                    }
                )
            # Sort by start_time
            scenes.sort(key=lambda x: x.get("start_time", 0))
            return scenes
        except Exception:
            return []

    def store_scene_metadata(self, media_path: str, scenes: list[dict]) -> None:
        """Stores scene-level metadata for a video.

        Note: Current implementation only logs the receipt of scenes.

        Args:
            media_path: Path to the source video.
            scenes: List of scene data dictionaries.
        """
        log(
            f"Received {len(scenes)} scenes for {media_path} (Storage not implemented)"
        )

    def get_masklets_for_media(self, media_path: str) -> list[dict]:
        """Retrieve all masklets (SAM tracks) for a specific video."""
        try:
            resp = self.client.scroll(
                collection_name=self.MASKLETS_COLLECTION,
                scroll_filter=models.Filter(
                    must=[
                        models.FieldCondition(
                            key="media_path",
                            match=models.MatchValue(value=media_path),
                        )
                    ]
                ),
                limit=1000,
                with_payload=True,
                with_vectors=False,
            )
            masklets = []
            for p in resp[0]:
                payload = p.payload or {}
                masklets.append({
                    "id": p.id,
                    "concept": payload.get("concept"),
                    "start_time": payload.get("start_time"),
                    "end_time": payload.get("end_time"),
                    "confidence": payload.get("confidence", 1.0),
                    "bbox": payload.get("bbox"),
                    "frame_idx": payload.get("frame_idx")
                })
            return masklets
        except Exception as e:
            log(f"get_masklets_for_media failed: {e}")
            return []

