"""Search repository - frame, media, and hybrid search operations.

Extracted from VectorDB to reduce God class size.
VectorDB inherits from SearchRepository to compose these methods.
"""

from __future__ import annotations

from core.domain.values import VideoPath, Timestamp, ClusterId, JobId

from typing import TYPE_CHECKING, Any

from qdrant_client.http import models

from core.utils.logger import log

if TYPE_CHECKING:
    from qdrant_client import QdrantClient


class SearchRepository:
    """Frame search, media search, hybrid search, and explainable search operations."""

    client: QdrantClient

    async def search_frames(
        self,
        query: str,
        limit: int = 10,
        score_threshold: float = None,  # Uses settings.frame_search_score_threshold if None
        allowed_video_paths: list[str] | None = None,
        face_cluster_id: int | None = None,
    ) -> list[dict[str, Any]]:
        """Search for frames using text query.

        Args:
            query: The search text.
            limit: Max results.
            score_threshold: Min similarity score.
            allowed_video_paths: Optional list of video paths to restrict search.
            face_cluster_id: Optional filter for specific face cluster.

        Returns:
            A list of payload dictionaries containing frame metadata.
        """
        # CRITICAL: Use Visual Encoder (SigLIP/CLIP) to encode text for Frame Search
        # Frames are indexed with Visual Encoder, so query MUST be in same latent space.
        # BGE is for TEXT-only collections - using it here would cause ~0 similarity.
        try:
            query_vector = await self.visual_encoder.encode_text(query)
            if not query_vector:
                log(
                    f"Visual encoder returned empty for query: '{query[:50]}'",
                    level="WARNING",
                )
                return []  # Return empty, not garbage results
        except Exception as e:
            log(
                f"Visual Text Encoding failed: {e}. Cannot search frames without visual encoder.",
                level="ERROR",
            )
            return []  # Return empty, not wrong-space results

        # Build filter conditions
        filter_conditions = []
        if allowed_video_paths:
            filter_conditions.append(
                models.FieldCondition(
                    key="video_path",
                    match=models.MatchAny(any=allowed_video_paths),
                )
            )

        if face_cluster_id is not None:
            filter_conditions.append(
                models.FieldCondition(
                    key="face_cluster_ids",
                    match=models.MatchValue(value=face_cluster_id),
                )
            )

        scroll_filter = None
        if filter_conditions:
            scroll_filter = models.Filter(
                must=cast(list[models.Condition], filter_conditions)
            )

        results = self.client.query_points(
            collection_name=self.MEDIA_COLLECTION,
            query=query_vector,
            query_filter=scroll_filter,
            limit=limit,
            score_threshold=score_threshold,
            with_payload=True,
        ).points
        return [point.payload for point in results if point.payload]

    async def search_media(
        self,
        query: str,
        limit: int = 5,
        score_threshold: float | None = None,
        video_path: str | None = None,
        segment_type: str | None = None,
    ) -> list[dict[str, Any]]:
        """Searches for media segments (dialogue/subtitles) similar to the query.

        Args:
            query: The search query string.
            limit: Maximum number of results to return.
            score_threshold: Minimum similarity score (0.0 to 1.0).
            video_path: If provided, restricts search to this specific video.
            segment_type: Filter by 'dialogue', 'subtitle', etc.

        Returns:
            A list of result dictionaries containing score, text, and timestamps.
        """
        query_vector = (await self.encode_texts(query, is_query=True))[0]

        conditions: list[models.Condition] = []
        if video_path:
            conditions.append(
                models.FieldCondition(
                    key="video_path",
                    match=models.MatchValue(value=video_path),
                )
            )
        if segment_type:
            conditions.append(
                models.FieldCondition(
                    key="type",
                    match=models.MatchValue(value=segment_type),
                )
            )

        qfilter = models.Filter(must=conditions) if conditions else None

        resp = self.client.query_points(
            collection_name=self.MEDIA_SEGMENTS_COLLECTION,
            query=query_vector,
            limit=limit,
            score_threshold=score_threshold,
            query_filter=qfilter,
        )

        results = []
        for hit in resp.points:
            payload = hit.payload or {}
            results.append(
                {
                    "score": hit.score,
                    "text": payload.get("text"),
                    "start": payload.get("start"),
                    "end": payload.get("end"),
                    "video_path": payload.get("video_path"),
                    "type": payload.get("type"),
                }
            )

        return results

    async def search_frames_filtered(
        self,
        query_vector: list[float] | str,
        face_cluster_ids: list[int] | None = None,
        limit: int = 20,
        score_threshold: float | None = None,
        video_path: str
        | None = None,  # CRITICAL: Prevent cross-video identity leakage
    ) -> list[dict[str, Any]]:
        """Search frames with optional identity and video filtering.

        Used by agentic search to filter by face_cluster_ids.
        IMPORTANT: Always pass video_path to prevent cross-video identity leakage.

        Args:
            query_vector: The query embedding vector OR natural language query string.
            face_cluster_ids: Face cluster IDs to filter by (identity filter).
            limit: Maximum number of results.
            score_threshold: Minimum similarity score.
            video_path: Filter results to this video only (prevents cross-video leakage).

        Returns:
            A list of matching frames with full payload.
        """
        # Auto-encode if string passed
        if isinstance(query_vector, str):
            query_vector = await self.get_embedding(query_vector)
            if query_vector is None:
                log("Embedding generation failed, cannot search frames", level="WARNING")
                return []

        # Build filter conditions
        conditions: list[models.Condition] = []

        # CRITICAL: Add video_path filter to prevent cross-video identity leakage
        if video_path:
            conditions.append(
                models.FieldCondition(
                    key="video_path",
                    match=models.MatchValue(value=video_path),
                )
            )

        # Face identity filter
        if face_cluster_ids:
            conditions.append(
                models.FieldCondition(
                    key="face_cluster_ids",
                    match=models.MatchAny(any=face_cluster_ids),
                )
            )

        query_filter = models.Filter(must=conditions) if conditions else None

        resp = self.client.query_points(
            collection_name=self.MEDIA_COLLECTION,
            query=query_vector,
            limit=limit,
            query_filter=query_filter,
            score_threshold=score_threshold,
        )

        results = []
        for hit in resp.points:
            payload = hit.payload or {}
            result = {
                "score": hit.score,
                "id": str(hit.id),
                **payload,  # Include all payload fields
            }
            results.append(result)

        return results

    async def search_frames_hybrid(
        self,
        query: str,
        limit: int = 20,
        video_paths: str | list[str] | None = None,
        face_cluster_ids: list[int] | None = None,
        rrf_k: int = None,  # Uses settings.rrf_constant if None
        transcript_query: str | None = None,
        music_section: str | None = None,
        high_energy: bool = False,
    ) -> list[dict[str, Any]]:
        """Performs a hybrid search combining vector, keyword, and identity filters."""
        from collections import defaultdict

        results_by_id: dict[str, dict] = {}
        rank_lists: dict[str, dict[str, int]] = defaultdict(dict)

        # Normalize video_paths to list
        if isinstance(video_paths, str):
            video_paths = [video_paths]

        # === 1. VECTOR SEARCH (Semantic Understanding) ===
        # CRITICAL: Frames are indexed with SigLIP visual embeddings.
        # Query MUST be encoded in the same visual space, not BGE text space.
        try:
            query_vector = None
            try:
                query_vector = await self.visual_encoder.encode_text(query)
            except Exception as ve:
                log(f"Visual encoder failed, falling back to text encoder: {ve}")
            if not query_vector:
                # Fallback: text encoder — may produce lower-quality results
                # but still better than no vector search at all
                await self._ensure_encoder_loaded()
                query_vector = (await self.encode_texts(query, is_query=True))[0]

            conditions = []
            if video_paths:
                conditions.append(
                    models.FieldCondition(
                        key="video_path",
                        match=models.MatchAny(any=video_paths),
                    )
                )
            if face_cluster_ids:
                conditions.append(
                    models.FieldCondition(
                        key="face_cluster_ids",
                        match=models.MatchAny(any=face_cluster_ids),
                    )
                )

            qfilter = models.Filter(must=conditions) if conditions else None

            vec_resp = self.client.query_points(
                collection_name=self.MEDIA_COLLECTION,
                query=query_vector,
                limit=limit * 2,
                query_filter=qfilter,
            )

            for rank, hit in enumerate(vec_resp.points):
                point_id = str(hit.id)
                rank_lists["vector"][point_id] = rank + 1
                if point_id not in results_by_id:
                    payload = hit.payload or {}
                    results_by_id[point_id] = {
                        "id": point_id,
                        "score": 0.0,
                        "vector_score": hit.score,
                        "match_reasons": [],
                        **payload,
                    }
                # Generate detailed match reason with actual content
                payload = hit.payload or {}
                desc_preview = (
                    payload.get("description") or payload.get("action") or ""
                )[:80]
                results_by_id[point_id]["match_reasons"].append(
                    f"Semantic match (score={hit.score:.2f}): {desc_preview}..."
                )
        except Exception as e:
            log(f"Vector search failed: {e}")

        log(f"Vector search found {len(results_by_id)} candidates so far")

        # === 2. KEYWORD SEARCH (Text Fields via Qdrant Indexes) ===
        try:
            # We use Qdrant's MatchText to find documents containing query terms.
            # This is much faster than scraping random frames.

            # Fields to search - COMPREHENSIVE list of ALL indexed text fields
            text_fields = [
                "action",
                "dialogue",
                "description",
                "entities",
                "visible_text",
                "face_names",
                "speaker_names",
                "visual_attributes",  # Dynamic: ALL visual details from VLM
                "entity_details",  # Dynamic: ALL entity names and categories
                "scene_location",
                "ocr_text",
                "transcript",
                "identity_text",
                "temporal_context",
            ]

            should_conditions = []
            for field in text_fields:
                should_conditions.append(
                    models.FieldCondition(
                        key=field, match=models.MatchText(text=query)
                    )
                )

            # Combine with strong filters (video path)
            must_conditions = []
            if video_paths:
                must_conditions.append(
                    models.FieldCondition(
                        key="video_path",
                        match=models.MatchAny(any=video_paths),
                    )
                )

            keyword_filter = models.Filter(
                should=should_conditions,
                must=must_conditions if must_conditions else None,
            )

            # Scroll for matches (limit to 100 high-relevance matches)
            # Since Qdrant basic text match doesn't score, we treat them as high-confidence hits.
            # Ideally we'd use sparse vectors for BM25, but this is a robust fallback.
            scroll_resp = self.client.scroll(
                collection_name=self.MEDIA_COLLECTION,
                scroll_filter=keyword_filter,
                limit=limit * 3,
                with_payload=True,
                with_vectors=False,
            )

            for rank, point in enumerate(scroll_resp[0]):
                point_id = str(point.id)
                rank_lists["keyword"][point_id] = rank + 1  # 1-based rank

                if point_id not in results_by_id:
                    payload = point.payload or {}
                    results_by_id[point_id] = {
                        "id": point_id,
                        "score": 0.0,
                        "keyword_score": 1.0,  # Placeholder info
                        "match_reasons": [],
                        **payload,
                    }

                # Try to determine WHY it matched for the UI
                payload = results_by_id[point_id]
                matched_fields = []
                q_lower = query.lower().split()

                # Heuristic verify
                for field in text_fields:
                    val = str(payload.get(field, "")).lower()
                    if any(w in val for w in q_lower if len(w) > 2):
                        matched_fields.append(field)

                # Generate detailed match reason showing WHAT matched
                field_details = []
                for field in matched_fields:
                    val = str(payload.get(field, ""))[:50]
                    if val:
                        field_details.append(f"{field}='{val}'")

                if field_details:
                    results_by_id[point_id]["match_reasons"].append(
                        f"Text match: {'; '.join(field_details[:3])}"
                    )
                else:
                    results_by_id[point_id]["match_reasons"].append(
                        "Text match (partial)"
                    )

            # === 2b. MASKLET SEARCH (Deep Video Understanding) ===
            # Search for SAM3-tracked concepts overlap with query
            try:
                masklet_conditions = []
                if video_paths:
                    masklet_conditions.append(
                        models.FieldCondition(
                            key="video_path",
                            match=models.MatchAny(any=video_paths),
                        )
                    )
                # Check for concept match
                # Use MatchText for concepts too
                masklet_conditions.append(
                    models.FieldCondition(
                        key="concept",
                        match=models.MatchText(text=query),
                    )
                )

                masklet_filter = models.Filter(must=masklet_conditions)

                mask_resp = self.client.scroll(
                    collection_name=self.MASKLETS_COLLECTION,  # Fixed: use constant
                    scroll_filter=masklet_filter,
                    limit=50,
                    with_payload=True,
                )

                for point in mask_resp[0]:
                    payload = point.payload or {}
                    point_id = str(point.id)  # Use unique ID

                    rank_lists["keyword"][point_id] = 1  # High rank

                    if point_id not in results_by_id:
                        results_by_id[point_id] = {
                            "id": point_id,
                            "score": 0.0,
                            "keyword_score": 0.9,  # High confidence
                            "match_reasons": [],
                            "timestamp": payload.get("start_time", 0.0),
                            "video_path": payload.get("video_path"),
                            "action": f"Tracked concept: {payload.get('concept')}",
                            "type": "masklet",
                        }
                    results_by_id[point_id]["match_reasons"].append(
                        f"Concept match: {payload.get('concept')}"
                    )
            except Exception as e:
                log(f"Masklet search failed: {e}")

        except Exception as e:
            log(f"Keyword search failed: {e}")

        log(f"Keyword search found {len(rank_lists['keyword'])} matches")

        # === 3. IDENTITY SEARCH (Face/Speaker Names) ===
        try:
            identity_names = self._extract_identity_names(query)
            if identity_names:
                for name in identity_names:
                    cluster_id = self.fuzzy_get_cluster_id_by_name(name)
                    if cluster_id is not None:
                        conditions = [
                            models.FieldCondition(
                                key="face_cluster_ids",
                                match=models.MatchAny(any=[cluster_id]),
                            )
                        ]
                        if video_paths:
                            conditions.append(
                                models.FieldCondition(
                                    key="video_path",
                                    match=models.MatchAny(any=video_paths),
                                )
                            )

                        identity_resp = self.client.scroll(
                            collection_name=self.MEDIA_COLLECTION,
                            scroll_filter=models.Filter(must=conditions),  # type: ignore
                            limit=limit * 2,
                            with_payload=True,
                        )

                        for rank, point in enumerate(identity_resp[0]):
                            point_id = str(point.id)
                            rank_lists["identity"][point_id] = rank + 1
                            if point_id not in results_by_id:
                                payload = point.payload or {}
                                results_by_id[point_id] = {
                                    "id": point_id,
                                    "score": 0.0,
                                    "identity_match": True,
                                    "match_reasons": [],
                                    **payload,
                                }
                            # Get face names from the result payload
                            face_names = results_by_id[point_id].get(
                                "face_names", []
                            )
                            confidence = (
                                "high"
                                if len(rank_lists.get("identity", {})) <= 5
                                else "medium"
                            )
                            # Better formatting for unknown faces
                            display_name = name
                            if (
                                not display_name
                                or display_name.lower().startswith("unknown")
                            ):
                                display_name = f"Person {cluster_id}"

                            results_by_id[point_id]["match_reasons"].append(
                                f"Face identity: '{display_name}' (cluster={cluster_id}, conf={confidence}, visible_faces={face_names})"
                            )
                            results_by_id[point_id]["matched_identity"] = (
                                display_name
                            )
        except Exception as e:
            log(f"Identity search failed: {e}")

        # === 4. VOICE/TRANSCRIPT SEARCH (Who said what) ===
        try:
            # Search voice_segments for dialogue matching query
            voice_conditions = []
            if video_paths:
                voice_conditions.append(
                    models.FieldCondition(
                        key="media_path",
                        match=models.MatchAny(any=video_paths),
                    )
                )
            # Text match on transcript
            voice_conditions.append(
                models.FieldCondition(
                    key="transcript",
                    match=models.MatchText(text=transcript_query or query),
                )
            )

            voice_resp = self.client.scroll(
                collection_name=self.VOICE_COLLECTION,
                scroll_filter=models.Filter(must=voice_conditions),
                limit=limit * 2,
                with_payload=True,
            )

            for rank, point in enumerate(voice_resp[0]):
                payload = point.payload or {}
                # Create composite ID linking to timestamp
                media_path = payload.get("media_path", "")
                start_time = payload.get("start", 0)

                # Find corresponding frame at this timestamp
                frame_filter = models.Filter(
                    must=[
                        models.FieldCondition(
                            key="video_path",
                            match=models.MatchValue(value=media_path),
                        ),
                        models.FieldCondition(
                            key="timestamp",
                            range=models.Range(
                                gte=start_time - 2, lte=start_time + 2
                            ),
                        ),
                    ]
                )

                frame_resp = self.client.scroll(
                    collection_name=self.MEDIA_COLLECTION,
                    scroll_filter=frame_filter,
                    limit=1,
                    with_payload=True,
                )

                if frame_resp[0]:
                    frame_point = frame_resp[0][0]
                    point_id = str(frame_point.id)
                    rank_lists["voice"][point_id] = rank + 1

                    if point_id not in results_by_id:
                        frame_payload = frame_point.payload or {}
                        results_by_id[point_id] = {
                            "id": point_id,
                            "score": 0.0,
                            "voice_match": True,
                            "match_reasons": [],
                            **frame_payload,
                        }

                    # Detailed voice/transcript match info
                    transcript = payload.get("transcript", "")[:100]
                    speaker = (
                        payload.get("speaker_name")
                        or f"Speaker #{payload.get('cluster_id', '?')}"
                    )
                    results_by_id[point_id]["match_reasons"].append(
                        f"Dialogue match: '{speaker}' said '{transcript}...'"
                    )
                    results_by_id[point_id]["matched_dialogue"] = transcript

        except Exception as e:
            log(f"Voice/transcript search failed: {e}")

        # === 4b. MEDIA SEGMENT SEARCH (ASR/Subtitles) ===
        # Fallback for when voice diarization (VOICE_COLLECTION) misses segments
        try:
            asr_conditions = []
            if video_paths:
                asr_conditions.append(
                    models.FieldCondition(
                        key="video_path",
                        match=models.MatchAny(any=video_paths),
                    )
                )

            asr_conditions.append(
                models.FieldCondition(
                    key="text",
                    match=models.MatchText(text=transcript_query or query),
                )
            )

            asr_resp = self.client.scroll(
                collection_name=self.MEDIA_SEGMENTS_COLLECTION,
                scroll_filter=models.Filter(must=asr_conditions),
                limit=limit * 2,
                with_payload=True,
            )

            for rank, point in enumerate(asr_resp[0]):
                payload = point.payload or {}
                media_path = payload.get("video_path", "")
                start_time = payload.get("start", 0)
                text = payload.get("text", "")[:100]

                # Find matching frame
                frame_filter = models.Filter(
                    must=[
                        models.FieldCondition(
                            key="video_path",
                            match=models.MatchValue(value=media_path),
                        ),
                        models.FieldCondition(
                            key="timestamp",
                            range=models.Range(
                                gte=start_time - 2, lte=start_time + 2
                            ),
                        ),
                    ]
                )

                frame_resp = self.client.scroll(
                    collection_name=self.MEDIA_COLLECTION,
                    scroll_filter=frame_filter,
                    limit=1,
                    with_payload=True,
                )

                if frame_resp[0]:
                    frame_point = frame_resp[0][0]
                    point_id = str(frame_point.id)

                    # Merge into voice rank list (treating ASR as voice)
                    if point_id not in rank_lists["voice"]:
                        rank_lists["voice"][point_id] = rank + 1
                    else:
                        rank_lists["voice"][point_id] = min(
                            rank_lists["voice"][point_id], rank + 1
                        )

                    if point_id not in results_by_id:
                        frame_payload = frame_point.payload or {}
                        results_by_id[point_id] = {
                            "id": point_id,
                            "score": 0.0,
                            "voice_match": True,
                            "match_reasons": [],
                            **frame_payload,
                        }

                    if (
                        f"Dialogue match: '{text}...'"
                        not in results_by_id[point_id]["match_reasons"]
                    ):
                        results_by_id[point_id]["match_reasons"].append(
                            f"Transcript match: '{text}...'"
                        )
                        results_by_id[point_id]["matched_dialogue"] = text

        except Exception as e:
            log(f"ASR/Media segment search failed: {e}")

        # === 5. AUDIO EVENT SEARCH (Music/Sounds) ===
        try:
            audio_conditions = []
            if video_paths:
                audio_conditions.append(
                    models.FieldCondition(
                        key="media_path",
                        match=models.MatchAny(any=video_paths),
                    )
                )
            # Match audio event type/label
            audio_conditions.append(
                models.FieldCondition(
                    key="event_type",
                    match=models.MatchText(text=query),
                )
            )

            audio_resp = self.client.scroll(
                collection_name=self.AUDIO_EVENTS_COLLECTION,
                scroll_filter=models.Filter(
                    should=audio_conditions
                ),  # OR condition
                limit=limit,
                with_payload=True,
            )

            for rank, point in enumerate(audio_resp[0]):
                payload = point.payload or {}
                media_path = payload.get("media_path", "")
                start_time = payload.get("start_time", 0)

                # Find frame near this audio event
                if media_path:
                    frame_filter = models.Filter(
                        must=[
                            models.FieldCondition(
                                key="video_path",
                                match=models.MatchValue(value=media_path),
                            ),
                            models.FieldCondition(
                                key="timestamp",
                                range=models.Range(
                                    gte=start_time - 1, lte=start_time + 3
                                ),
                            ),
                        ]
                    )

                    frame_resp = self.client.scroll(
                        collection_name=self.MEDIA_COLLECTION,
                        scroll_filter=frame_filter,
                        limit=1,
                        with_payload=True,
                    )

                    if frame_resp[0]:
                        frame_point = frame_resp[0][0]
                        point_id = str(frame_point.id)
                        rank_lists["audio"][point_id] = rank + 1

                        if point_id not in results_by_id:
                            frame_payload = frame_point.payload or {}
                            results_by_id[point_id] = {
                                "id": point_id,
                                "score": 0.0,
                                "audio_event_match": True,
                                "match_reasons": [],
                                **frame_payload,
                            }

                        # Detailed audio event info
                        event_type = payload.get("event_type", "unknown")
                        confidence = payload.get("confidence", 0)
                        results_by_id[point_id]["match_reasons"].append(
                            f"Audio event: '{event_type}' (conf={confidence:.2f}) at {start_time:.1f}s"
                        )

        except Exception as e:
            log(f"Audio event search failed: {e}")

        # === 5b. MUSIC SECTION FILTERING (Temporal precision) ===
        # Filter for specific music sections like "chorus" or "drop"
        if music_section:
            try:
                section_event = f"music_{music_section.lower()}"
                section_conditions = [
                    models.FieldCondition(
                        key="event_type",
                        match=models.MatchValue(value=section_event),
                    )
                ]
                if video_paths:
                    section_conditions.append(
                        models.FieldCondition(
                            key="media_path",
                            match=models.MatchAny(any=video_paths),
                        )
                    )

                section_resp = self.client.scroll(
                    collection_name=self.AUDIO_EVENTS_COLLECTION,
                    scroll_filter=models.Filter(must=section_conditions),
                    limit=50,
                    with_payload=True,
                )

                # Get time ranges for this section type
                section_ranges = []
                for point in section_resp[0]:
                    payload = point.payload or {}
                    start = payload.get("start_time", 0)
                    end = payload.get(
                        "end_time", start + 30
                    )  # Default 30s if no end
                    media = payload.get("media_path", "")
                    section_ranges.append((media, start, end))

                if section_ranges:
                    log(
                        f"[Music] Found {len(section_ranges)} '{music_section}' sections"
                    )

                    # Boost results that fall within these time ranges
                    for point_id, result in results_by_id.items():
                        video_path = result.get("video_path", "")
                        timestamp = result.get("timestamp", 0)

                        for media, start, end in section_ranges:
                            if (
                                video_path == media
                                and start <= timestamp <= end
                            ):
                                rank_lists["music_section"][point_id] = (
                                    1  # High rank
                                )
                                result["match_reasons"].append(
                                    f"During {music_section} ({start:.1f}s-{end:.1f}s)"
                                )
                                result["in_music_section"] = music_section
                                break

            except Exception as e:
                log(f"Music section filtering failed: {e}")

        # === 5c. HIGH ENERGY FILTERING ===
        if high_energy:
            try:
                # Filter for high-energy moments (drops, choruses)
                energy_conditions = []
                if video_paths:
                    energy_conditions.append(
                        models.FieldCondition(
                            key="media_path",
                            match=models.MatchAny(any=video_paths),
                        )
                    )

                # Look for drops and choruses (typically high energy)
                energy_resp = self.client.scroll(
                    collection_name=self.AUDIO_EVENTS_COLLECTION,
                    scroll_filter=models.Filter(
                        must=energy_conditions,
                        should=[
                            models.FieldCondition(
                                key="event_type",
                                match=models.MatchValue(value="music_drop"),
                            ),
                            models.FieldCondition(
                                key="event_type",
                                match=models.MatchValue(value="music_chorus"),
                            ),
                        ],
                    )
                    if energy_conditions
                    else models.Filter(
                        should=[
                            models.FieldCondition(
                                key="event_type",
                                match=models.MatchValue(value="music_drop"),
                            ),
                            models.FieldCondition(
                                key="event_type",
                                match=models.MatchValue(value="music_chorus"),
                            ),
                        ]
                    ),
                    limit=50,
                    with_payload=True,
                )

                energy_ranges = []
                for point in energy_resp[0]:
                    payload = point.payload or {}
                    energy_level = payload.get("payload", {}).get("energy", 0)
                    if energy_level > 0.8:  # Only high-energy sections
                        start = payload.get("start_time", 0)
                        end = payload.get("end_time", start + 30)
                        media = payload.get("media_path", "")
                        energy_ranges.append((media, start, end))

                if energy_ranges:
                    log(
                        f"[Energy] Found {len(energy_ranges)} high-energy sections"
                    )

                    for point_id, result in results_by_id.items():
                        video_path = result.get("video_path", "")
                        timestamp = result.get("timestamp", 0)

                        for media, start, end in energy_ranges:
                            if (
                                video_path == media
                                and start <= timestamp <= end
                            ):
                                rank_lists["high_energy"][point_id] = 1
                                result["match_reasons"].append(
                                    f"High-energy moment ({start:.1f}s-{end:.1f}s)"
                                )
                                result["is_high_energy"] = True
                                break

            except Exception as e:
                log(f"High energy filtering failed: {e}")

        log(
            "Total modalities searched: vector, keyword, identity, voice, audio, music_structure"
        )

        # === 6. RRF FUSION ===
        for point_id, result in results_by_id.items():
            rrf_score = 0.0
            for _method, ranks in rank_lists.items():
                if point_id in ranks:
                    rrf_score += 1.0 / (rrf_k + ranks[point_id])
            result["score"] = rrf_score
            result["rrf_score"] = rrf_score

        final_results = sorted(
            results_by_id.values(),
            key=lambda x: x["score"],
            reverse=True,
        )[:limit]

        return final_results

    def _extract_identity_names(self, query: str) -> list[str]:
        """Extract potential person names from query for identity search."""
        known_names = set()
        try:
            resp = self.client.scroll(
                collection_name=self.FACES_COLLECTION,
                limit=500,
                with_payload=["name"],
                with_vectors=False,
            )
            for pt in resp[0]:
                name = (pt.payload or {}).get("name")
                if name:
                    known_names.add(name.lower())
        except Exception as e:
            log(f"extract_names_from_query scroll failed: {e}", level="DEBUG")

        query_lower = query.lower()
        found_names = []
        for name in known_names:
            if name in query_lower:
                found_names.append(name)

        return found_names

    async def search_audio_events(
        self,
        query: str,
        limit: int = 10,
        score_threshold: float | None = None,
        video_path: str | None = None,
    ) -> list[dict[str, Any]]:
        conditions = []
        conditions.append(
            models.FieldCondition(
                key="event_class",
                match=models.MatchText(text=query),
            )
        )
        if video_path:
            conditions.append(
                models.FieldCondition(
                    key="media_path",
                    match=models.MatchValue(value=video_path),
                )
            )

        try:
            resp, _ = self.client.scroll(
                collection_name=self.AUDIO_EVENTS_COLLECTION,
                scroll_filter=models.Filter(must=conditions),
                limit=limit,
            )
            return [
                {"id": str(p.id), "score": 1.0, **(p.payload or {})}
                for p in resp
            ]
        except Exception as e:
            log(f"Audio event search failed: {e}")
            return []

    async def search_audio_events_semantic(
        self,
        query: str,
        limit: int = 10,
        score_threshold: float | None = None,
        video_path: str | None = None,
    ) -> list[dict[str, Any]]:
        """Search audio events using semantic vector similarity via CLAP.

        Uses CLAP text encoder to match against stored CLAP audio embeddings.
        Unlike search_audio_events which does text matching on event_class,
        this method performs vector-based semantic search for more flexible
        audio event discovery (e.g., "sudden loud noise" can match "explosion").

        Args:
            query: The search query.
            limit: Maximum number of results.
            score_threshold: Minimum similarity score.
            video_path: Optional filter by video path.

        Returns:
            List of matching audio events with scores.
        """
        try:
            # Use CLAP text encoder to get 512-dim embedding for audio search
            from core.processing.audio_events import get_audio_detector

            audio_detector = get_audio_detector()
            query_vec = await audio_detector.encode_text(query)

            if query_vec is None:
                log(
                    "CLAP text encoder unavailable, falling back to text search"
                )
                # Fallback to text-based search
                return await self.search_audio_events(
                    query=query,
                    limit=limit,
                    video_path=video_path,
                )

            conditions: list[models.Condition] = []
            # Only search events with actual embeddings
            conditions.append(
                models.FieldCondition(
                    key="has_embedding",
                    match=models.MatchValue(value=True),
                )
            )
            if video_path:
                conditions.append(
                    models.FieldCondition(
                        key="media_path",
                        match=models.MatchValue(value=video_path),
                    )
                )

            query_filter = (
                models.Filter(must=conditions) if conditions else None
            )

            resp = self.client.query_points(
                collection_name=self.AUDIO_EVENTS_COLLECTION,
                query=query_vec,
                limit=limit,
                score_threshold=score_threshold,
                query_filter=query_filter,
            )

            results = []
            for hit in resp.points:
                payload = hit.payload or {}
                results.append(
                    {
                        "id": str(hit.id),
                        "score": hit.score,
                        "type": "audio_event",
                        "label": payload.get(
                            "event", payload.get("label", "audio")
                        ),
                        "start": payload.get("start_time", 0),
                        "end": payload.get("end_time", 0),
                        "video_path": payload.get("media_path"),
                        "confidence": payload.get("confidence", 0),
                        **payload,
                    }
                )
            return results
        except Exception as e:
            log(f"search_audio_events_semantic failed: {e}")
            return []

    def explainable_search(
        self,
        query_text: str,
        parsed_query: Any = None,
        limit: int = 10,
        score_threshold: float = 0.3,
    ) -> list[dict[str, Any]]:
        """Search with explainable results - returns reasoning for each match.

        This is the SOTA search method that provides:
        - Matched entities with individual confidence scores
        - Reasoning for why the result was selected
        - Face/voice identification with names
        - Timestamp accuracy justification

        Args:
            query_text: The search query text.
            parsed_query: Optional DynamicParsedQuery with extracted entities.
            limit: Maximum results to return.
            score_threshold: Minimum similarity score.

        Returns:
            List of results with explainable metadata:
            [
                {
                    "id": "...",
                    "score": 0.85,
                    "timestamp": 45.2,
                    "segment_url": "/media/segment?...",
                    "matched_entities": {
                        "person": {"name": "Prakash", "confidence": 0.98, "source": "face"},
                        "clothing": [{"item": "blue t-shirt", "confidence": 0.87}],
                        "action": {"name": "bowling", "confidence": 0.92}
                    },
                    "reasoning": "Frame shows Prakash (face ID #3) in blue upper garment...",
                    "evidence": ["face_match", "color_match", "action_match"]
                }
            ]
        """
        # 1. Generate embedding for query
        self.get_embedding(query_text)

        # 2. Perform multi-vector search
        raw_results = self.search_scenes(
            query=query_text,
            limit=limit * 2,  # Get more candidates for re-ranking
            score_threshold=score_threshold,
            search_mode="hybrid",
        )

        # 3. Enrich each result with explainable metadata
        explainable_results = []

        for result in raw_results[:limit]:
            # Extract entities from the result payload
            matched_entities = {}
            evidence = []
            reasoning_parts = []

            # Check for person/face matches
            face_names = result.get("face_names", []) or result.get(
                "person_names", []
            )
            face_ids = result.get("face_ids", [])
            if face_names:
                for i, name in enumerate(face_names):
                    if name:
                        matched_entities["person"] = {
                            "name": name,
                            "confidence": 0.95,  # Face recognition typically high confidence
                            "source": "face_recognition",
                            "face_id": face_ids[i]
                            if i < len(face_ids)
                            else None,
                        }
                        evidence.append("face_match")
                        reasoning_parts.append(
                            f"Identified {name} via face recognition"
                        )
                        break

            # Check for voice matches
            voice_names = result.get("voice_names", []) or result.get(
                "speaker_names", []
            )
            voice_ids = result.get("voice_ids", [])
            if voice_names:
                for i, name in enumerate(voice_names):
                    if name:
                        matched_entities["voice"] = {
                            "name": name,
                            "confidence": 0.85,
                            "source": "voice_diarization",
                            "voice_id": voice_ids[i]
                            if i < len(voice_ids)
                            else None,
                        }
                        evidence.append("voice_match")
                        reasoning_parts.append(f"Voice identified as {name}")
                        break

            # Check for text/OCR matches
            visible_text = result.get("visible_text", []) or result.get(
                "ocr_text", []
            )
            if visible_text:
                matched_entities["text"] = {
                    "items": visible_text[:5],  # Top 5 text items
                    "confidence": 0.90,
                    "source": "ocr",
                }
                evidence.append("text_match")
                reasoning_parts.append(
                    f"Visible text: {', '.join(visible_text[:3])}"
                )

            # Check for location
            location = result.get("location", "") or result.get(
                "scene_location", ""
            )
            if location:
                matched_entities["location"] = {
                    "name": location,
                    "confidence": 0.80,
                    "source": "scene_analysis",
                }
                evidence.append("location_match")
                reasoning_parts.append(f"Location: {location}")

            # Check for actions
            actions = result.get("actions", []) or result.get(
                "action_keywords", []
            )
            if actions:
                matched_entities["actions"] = {
                    "items": actions[:5],
                    "confidence": 0.75,
                    "source": "visual_analysis",
                }
                evidence.append("action_match")
                reasoning_parts.append(f"Actions: {', '.join(actions[:3])}")

            # Get description for additional context
            description = result.get("description", "") or result.get(
                "dense_caption", ""
            )

            # Build reasoning string
            reasoning = (
                "; ".join(reasoning_parts)
                if reasoning_parts
                else description[:200]
            )

            # Build explainable result
            explainable_results.append(
                {
                    "id": result.get("id"),
                    "score": result.get("score", 0),
                    "timestamp": result.get("start_time")
                    or result.get("timestamp", 0),
                    "end_time": result.get("end_time"),
                    "media_path": result.get("media_path"),
                    "matched_entities": matched_entities,
                    "reasoning": reasoning,
                    "evidence": evidence,
                    "hitl_boost": result.get("hitl_boost", False),
                    # Include raw data for debugging
                    "raw_description": description[:500]
                    if description
                    else None,
                }
            )

        return explainable_results

    async def search_frames_hybrid_legacy(
        self,
        query: str,
        video_paths: str | list[str] | None = None,
        limit: int = 20,
        weights: dict[str, float] | None = None,
    ) -> list[dict[str, Any]]:
        """Legacy hybrid search with keyword boosting.

        NOTE: Main search_frames_hybrid is at line ~826 using RRF algorithm.
        This version uses simpler keyword boosting approach.

        Args:
            query: Natural language search query.
            video_paths: Optional filter to specific video(s).
            limit: Maximum results to return.
            weights: Optional dictionary of boosting weights.

        Returns:
            Ranked list of matching frames with scores.
        """
        # Default Weights (Tuned for balanced precision/recall)
        w = {
            "face_match": 0.20,
            "speaker_match": 0.15,
            "entity_match": 0.10,
            "text_match": 0.08,
            "scene_match": 0.08,
            "action_match": 0.05,
        }
        if weights:
            w.update(weights)

        # 1. Check for HITL names in query
        known_names = self.get_all_hitl_names()
        query_lower = query.lower()
        matched_names = [n for n in known_names if n.lower() in query_lower]

        identity_filter = None
        if matched_names:
            # Get cluster IDs for matched names
            cluster_ids = []
            for name in matched_names:
                cid = self.get_cluster_id_by_name(name)
                if cid is not None:
                    cluster_ids.append(cid)

            if cluster_ids:
                identity_filter = models.Filter(
                    must=[
                        models.FieldCondition(
                            key="face_cluster_ids",
                            match=models.MatchAny(any=cluster_ids),
                        )
                    ]
                )
                log(
                    f"[HybridSearch] Identity filter: {matched_names} → clusters {cluster_ids}"
                )

        # 2. Build video path filter
        video_filter = None
        if video_paths:
            if isinstance(video_paths, str):
                video_paths = [video_paths]
            video_filter = models.FieldCondition(
                key="video_path",
                match=models.MatchAny(any=video_paths),
            )

        # Combine filters
        combined_filter = None
        conditions = []
        if identity_filter:
            conditions.extend(identity_filter.must or [])
        if video_filter:
            conditions.append(video_filter)
        if conditions:
            combined_filter = models.Filter(must=conditions)

        # 3. Vector search
        query_vector = (await self.encode_texts(query, is_query=True))[0]

        vector_results = self.client.query_points(
            collection_name=self.MEDIA_COLLECTION,
            query=query_vector,
            limit=limit * 3,
            query_filter=combined_filter,
        )

        # 4. Extract keywords for boosting
        query_words = {w.lower() for w in query.split() if len(w) > 2}
        # Remove common words
        stopwords = {
            "the",
            "and",
            "for",
            "with",
            "that",
            "this",
            "from",
            "are",
            "was",
            "were",
        }
        query_words -= stopwords

        # 5. Score and boost results
        results = []
        for hit in vector_results.points:
            payload = hit.payload or {}
            score = float(hit.score or 0)

            # Keyword boost on structured fields
            boost = 0.0

            # Check face_names
            for name in payload.get("face_names", []):
                if name and name.lower() in query_lower:
                    boost += w["face_match"]

            # Check speaker_names
            for name in payload.get("speaker_names", []):
                if name and name.lower() in query_lower:
                    boost += w["speaker_match"]

            # Check entities
            for entity in payload.get("entities", []):
                if entity and any(w in entity.lower() for w in query_words):
                    boost += w["entity_match"]

            # Check visible_text
            for text in payload.get("visible_text", []):
                if text and any(w in text.lower() for w in query_words):
                    boost += w["text_match"]

            # Check scene_location
            location = payload.get("scene_location", "") or ""
            if any(w in location.lower() for w in query_words):
                boost += w["scene_match"]

            # Check action/description
            action = (
                payload.get("action", "")
                or payload.get("description", "")
                or ""
            )
            if any(w in action.lower() for w in query_words):
                boost += w["action_match"]

            results.append(
                {
                    "id": str(hit.id),
                    "score": score + boost,
                    "base_score": score,
                    "keyword_boost": boost,
                    **payload,
                }
            )

        # 6. Sort by final score
        results.sort(key=lambda x: x["score"], reverse=True)

        log(
            f"[HybridSearch] Query: '{query}' | Identity filter: {bool(identity_filter)} | Results: {len(results)}"
        )

        return results[:limit]

