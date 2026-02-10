"""Agentic Search — orchestrator that composes query parsing and result processing.

Uses mixin inheritance to separate concerns:
  - QueryParserMixin: parsing, caching, identity resolution
  - ResultProcessorMixin: reranking, granular scoring, RRF fusion
"""

from __future__ import annotations

import asyncio
from collections import defaultdict
from typing import TYPE_CHECKING, Any

from core.knowledge.schemas import ParsedQuery
from core.retrieval.query_parser import QueryParserMixin
from core.retrieval.reranker import RerankingCouncil
from core.retrieval.result_processor import ResultProcessorMixin
from core.utils.logger import log
from core.utils.observe import observe
from core.utils.prompt_loader import load_prompt

from llm.factory import LLMFactory
from config import settings

if TYPE_CHECKING:
    from core.storage.db import VectorDB
    from llm.interface import LLMInterface


class SearchAgent(QueryParserMixin, ResultProcessorMixin):

    def __init__(
        self,
        db: VectorDB,
        llm: LLMInterface | None = None,
        enable_hybrid: bool = True,
        enable_graph: bool = True,
    ) -> None:
        self.db = db
        self.llm = llm or LLMFactory.create_llm()
        self._hybrid_searcher = None
        self._enable_hybrid = enable_hybrid
        self._enable_graph = enable_graph
        self._council = None
        self._graph_searcher = None
        self._init_cache()

    # ----- lazy properties -----

    @property
    def hybrid_searcher(self):
        if self._hybrid_searcher is None and self._enable_hybrid:
            try:
                from core.retrieval.hybrid import HybridSearcher
                self._hybrid_searcher = HybridSearcher(self.db)
                log("[Search] HybridSearcher initialized")
            except Exception as e:
                log(f"[Search] HybridSearcher init failed: {e}")
        return self._hybrid_searcher

    @property
    def council(self) -> RerankingCouncil:
        if self._council is None:
            self._council = RerankingCouncil()
        return self._council

    @property
    def graph_searcher(self):
        if self._graph_searcher is None and self._enable_graph:
            try:
                from core.retrieval.graph import GraphSearcher
                self._graph_searcher = GraphSearcher()
                log("[Search] GraphSearcher initialized - PROD READY")
            except Exception as e:
                log(f"[Search] GraphSearcher init failed: {e}")
        return self._graph_searcher

    # =========================================================================
    # SEARCH METHODS
    # =========================================================================

    @observe("search_scenes_agentic")
    async def search_scenes(
        self,
        query: str,
        limit: int = 20,
        use_expansion: bool = False,
        video_path: str | None = None,
    ) -> dict[str, Any]:
        log(f"[Search] Scene search: '{query[:100]}...'")
        results = []

        if use_expansion:
            parsed = await self.parse_query(query)
        else:
            parsed = ParsedQuery(visual_keywords=[query])

        cluster_id: int | None = None
        face_ids: list[str] = []
        resolved_name: str | None = None

        if parsed.person_name:
            cluster_id = self._resolve_identity(parsed.person_name)
            if cluster_id is not None:
                face_ids = self._get_face_ids_for_cluster(cluster_id)
                resolved_name = parsed.person_name
                log(f"[Search] Resolved '{parsed.person_name}' → cluster {cluster_id}")

            masklet_hits = await self.db.search_masklets(
                concept=parsed.person_name, limit=5
            )
            if masklet_hits:
                log(f"[Search] Found {len(masklet_hits)} tagged masklets for '{parsed.person_name}'")
                for hit in masklet_hits:
                    results.append({
                        "id": hit.get("id"),
                        "score": hit.get("score", 1.0) * 1.5,
                        "type": "object_track",
                        "text": f"Tracked Object: {parsed.person_name}",
                        "start": hit.get("start_time"),
                        "end": hit.get("end_time"),
                        "video_path": hit.get("video_path"),
                        "thumbnail_url": f"/thumbnails/{hit.get('id')}.jpg"
                    })

        search_text = parsed.to_search_text()
        log(f"[Search] Expanded search text: '{search_text}'")

        try:
            scene_results = await self.db.search_scenes(
                query=search_text,
                limit=limit,
                person_name=resolved_name,
                face_cluster_ids=[cluster_id] if cluster_id is not None else None,
                clothing_color=parsed.clothing_color,
                clothing_type=parsed.clothing_type,
                accessories=parsed.accessories if parsed.accessories else None,
                location=parsed.location,
                visible_text=parsed.text_to_find if parsed.text_to_find else None,
                action_keywords=parsed.action_keywords if parsed.action_keywords else None,
                video_path=video_path,
                mood=parsed.mood,
                shot_type=parsed.shot_type,
                aesthetic_score=parsed.aesthetic_score,
                search_mode="hybrid",
            )
            results.extend(scene_results)
            log(f"[Search] Found {len(scene_results)} scene results")
        except Exception as e:
            log(f"[Search] Scene search failed: {e}, falling back to frame search")
            frame_results = await self._fallback_frame_search(parsed, search_text, limit)
            results.extend(frame_results)

        if not results:
            log("[Search] No scene results found. Triggering fallback frame search.")
            frame_results = await self._fallback_frame_search(parsed, search_text, limit)
            results.extend(frame_results)

        # Graph search
        if self.graph_searcher:
            try:
                graph_entities = []
                if parsed.person_name:
                    graph_entities.append(parsed.person_name)
                    if cluster_id is not None:
                        graph_entities.append(str(cluster_id))
                if parsed.visual_keywords:
                    graph_entities.extend([k for k in parsed.visual_keywords if len(k) > 3])

                graph_actions = parsed.action_keywords if parsed.action_keywords else []

                if graph_entities or graph_actions:
                    log(f"[Search] Executing Graph Query for entities={graph_entities}, actions={graph_actions}")
                    graph_results = await self.graph_searcher.search(
                        query=query, entities=graph_entities, actions=graph_actions
                    )
                    if graph_results:
                        log(f"[Search] Graph found {len(graph_results)} results. Merging...")
                        existing_ids = {r.get("id") for r in results}
                        for gr in graph_results:
                            if gr.get("id") not in existing_ids:
                                gr_formatted = {
                                    "id": gr.get("id"),
                                    "score": gr.get("score", 0.9),
                                    "type": "scene_graph_match",
                                    "text": f"Graph Match: {gr.get('description', '')}",
                                    "start_time": gr.get("start"),
                                    "end_time": gr.get("end"),
                                    "video_path": gr.get("video_path") or video_path,
                                }
                                if not gr_formatted.get("video_path") and len(results) > 0:
                                    gr_formatted["video_path"] = results[0].get("video_path")
                                results.append(gr_formatted)
            except Exception as e:
                log(f"[Search] Graph search failed: {e}")

        # Enrich results with UI overlay data
        for res in results:
            try:
                vid_path = res.get("video_path")
                start = res.get("start_time")
                end = res.get("end_time") or (start + 2.0 if start else None)
                if vid_path and start is not None and end is not None:
                    res["face_bboxes"] = self.db.get_faces_in_range(vid_path, start, end)
                    res["voice_segments"] = self.db.get_voice_segments_in_range(vid_path, start, end)
            except Exception as e:
                log(f"[Search] Failed to enrich result {res.get('id')}: {e}")

        reasoning_chain = {
            "step1_parse": f"Parsed '{query[:50]}...' into structured query",
            "step2_identity": f"Resolved identity: {resolved_name or 'None'} → cluster {cluster_id}",
            "step3_expand": f"Expanded to: {search_text[:80]}...",
            "step4_filters": {
                "face_cluster": cluster_id,
                "clothing_color": parsed.clothing_color,
                "clothing_type": parsed.clothing_type,
                "accessories": parsed.accessories,
                "location": parsed.location,
                "visible_text": parsed.text_to_find,
                "video_path": video_path,
            },
            "step5_results": f"Found {len(results)} scenes",
        }
        top_scores = [
            {r.get("id", "?"): round(r.get("score", 0), 3)} for r in results[:5]
        ]
        log(f"[Search] Original: {query}")
        log(f"[Search] Expanded: {search_text}")
        log(f"[Search] Filters: faces={[cluster_id] if cluster_id else []}, video={video_path or 'all'}")
        log(f"[Search] Scoring: {top_scores}")
        log(f"[Search] Reasoning: {reasoning_chain}")

        return {
            "query": query,
            "parsed": parsed.model_dump(),
            "resolved_identity": resolved_name,
            "face_ids_matched": len(face_ids),
            "expanded_search": search_text,
            "results": results,
            "result_count": len(results),
            "search_type": "scene",
            "reasoning_chain": reasoning_chain,
        }

    @observe("search_agentic")
    async def search(
        self,
        query: str,
        limit: int = 20,
        use_expansion: bool = False,
    ) -> dict[str, Any]:
        log(f"[Search] Agentic frame search: '{query}'")

        if use_expansion:
            parsed = await self.parse_query(query)
        else:
            parsed = ParsedQuery(visual_keywords=[query])

        face_ids: list[str] = []
        resolved_name: str | None = None
        cluster_id: int | None = None

        if parsed.person_name:
            cluster_id = self._resolve_identity(parsed.person_name)
            if cluster_id is not None:
                face_ids = self._get_face_ids_for_cluster(cluster_id)
                resolved_name = parsed.person_name
                log(f"[Search] Resolved '{parsed.person_name}' → cluster {cluster_id} ({len(face_ids)} faces)")

        search_text = parsed.to_search_text()
        log(f"[Search] Expanded search text: '{search_text}'")

        from qdrant_client.http import models

        filters: list[models.FieldCondition] = []

        if cluster_id is not None:
            filters.append(
                models.FieldCondition(
                    key="face_cluster_ids",
                    match=models.MatchAny(any=[cluster_id]),
                )
            )

        if parsed.clothing_color:
            filters.append(
                models.FieldCondition(
                    key="clothing_colors",
                    match=models.MatchAny(any=[parsed.clothing_color.lower()]),
                )
            )

        if parsed.clothing_type:
            filters.append(
                models.FieldCondition(
                    key="clothing_types",
                    match=models.MatchAny(any=[parsed.clothing_type.lower()]),
                )
            )

        if parsed.accessories:
            filters.append(
                models.FieldCondition(
                    key="accessories",
                    match=models.MatchAny(any=parsed.accessories),
                )
            )

        if parsed.text_to_find:
            filters.append(
                models.FieldCondition(
                    key="visible_text",
                    match=models.MatchAny(any=parsed.text_to_find),
                )
            )

        if parsed.location:
            filters.append(
                models.FieldCondition(
                    key="scene_location",
                    match=models.MatchText(text=parsed.location),
                )
            )

        if parsed.visual_keywords:
            specific_entities = [k for k in parsed.visual_keywords if len(k) > 3]
            if specific_entities:
                filters.append(
                    models.FieldCondition(
                        key="entity_names",
                        match=models.MatchAny(any=specific_entities),
                    )
                )

        try:
            query_vector = (
                await self.db.encode_texts(search_text or "scene activity", is_query=True)
            )[0]

            if filters:
                conditions: list[models.Condition] = list(filters)
                results = self.db.client.query_points(
                    collection_name=self.db.MEDIA_COLLECTION,
                    query=query_vector,
                    query_filter=models.Filter(should=conditions)
                    if len(conditions) > 1
                    else models.Filter(must=conditions),
                    limit=limit,
                ).points
                results = [
                    {"score": hit.score, "id": str(hit.id), **(hit.payload or {})}
                    for hit in results
                ]
            else:
                results = await self.db.search_frames(query=search_text, limit=limit)
        except Exception as e:
            log(f"[Search] Hybrid search failed: {e}, falling back to simple")
            results = await self.db.search_frames(query=search_text, limit=limit)

        return {
            "query": query,
            "parsed": parsed.model_dump(),
            "resolved_identity": resolved_name,
            "face_ids_matched": len(face_ids),
            "expanded_search": search_text,
            "results": results,
            "result_count": len(results),
            "search_type": "frame",
        }

    async def _fallback_frame_search(
        self, parsed: ParsedQuery, search_text: str, limit: int,
    ) -> list[dict]:
        try:
            return await self.db.search_frames(query=search_text, limit=limit)
        except Exception:
            return []

    async def search_simple(self, query: str, limit: int = 20) -> list[dict]:
        return await self.db.search_frames(query=query, limit=limit)

    async def hybrid_search(
        self,
        query: str,
        limit: int = 50,
        vector_weight: float = 0.7,
        keyword_weight: float = 0.3,
        video_id: str | None = None,
    ) -> list[dict]:
        if self.hybrid_searcher:
            try:
                results = await self.hybrid_searcher.search(
                    query=query, limit=limit,
                    vector_weight=vector_weight, keyword_weight=keyword_weight,
                    video_id=video_id,
                )
                log(f"[Search] Hybrid search: {len(results)} results (weights: {vector_weight:.1f}v/{keyword_weight:.1f}kw)")
                return results
            except Exception as e:
                log(f"[Search] Hybrid search failed, falling back: {e}")
        return await self.db.search_frames(query=query, limit=limit)

    # =========================================================================
    # SOTA SEARCH
    # =========================================================================

    @observe("search_sota")
    async def sota_search(
        self,
        query: str,
        limit: int = 10,
        video_path: str | None = None,
        use_reranking: bool = True,
        use_expansion: bool = False,
        expansion_fallback: bool = True,
    ) -> dict[str, Any]:
        log(f"[SOTA Search] Query: '{query[:100]}...'")
        log(f"[SOTA Search] Options: expansion={use_expansion}, fallback={expansion_fallback}, rerank={use_reranking}")

        # 1. Parse
        expansion_used = False
        try:
            parsed = await self.parse_query(query)
            if use_expansion:
                search_text = parsed.to_search_text() or query
                expansion_used = search_text != query
                log(f"[SOTA Search] Expanded: '{search_text[:100]}...'")
            else:
                search_text = query
                log("[SOTA Search] Expansion disabled, using raw query with parsed constraints")
        except Exception as e:
            log(f"[SOTA Search] Parse failed: {e}, using raw query")
            parsed = ParsedQuery(visual_keywords=[query])
            search_text = query

        # 2. Resolve identities
        person_names = []
        face_ids = []

        if hasattr(parsed, "entities") and parsed.entities:
            for entity in parsed.entities:
                if entity.entity_type.lower() == "person" and entity.name:
                    person_names.append(entity.name)
        elif parsed.person_name:
            person_names.append(parsed.person_name)

        for name in person_names:
            cluster_id = self._resolve_identity(name)
            if cluster_id:
                ids = self._get_face_ids_for_cluster(cluster_id)
                face_ids.extend(ids)
                log(f"[SOTA Search] Resolved '{name}' → {len(ids)} faces")

        all_results = {}

        # Extract clothing from parsed constraints
        all_clothing_colors: list[str] = []
        all_clothing_types: list[str] = []
        all_accessories: list[str] = list(parsed.accessories) if hasattr(parsed, "accessories") else []

        if hasattr(parsed, "clothing") and parsed.clothing:
            for c in parsed.clothing:
                if c.get("color"):
                    all_clothing_colors.append(c["color"].lower())
                if c.get("item"):
                    all_clothing_types.append(c["item"].lower())
        if hasattr(parsed, "people") and parsed.people:
            for person in parsed.people:
                for ci in getattr(person, "clothing_items", []):
                    if hasattr(ci, "color") and ci.color:
                        all_clothing_colors.append(ci.color.lower())
                    if hasattr(ci, "item_type") and ci.item_type:
                        all_clothing_types.append(ci.item_type.lower())
        if not all_clothing_colors and parsed.clothing_color:
            all_clothing_colors.append(parsed.clothing_color.lower())
        if not all_clothing_types and parsed.clothing_type:
            all_clothing_types.append(parsed.clothing_type.lower())

        # 3. Multi-modality retrieval
        try:
            scene_results = await self.db.search_scenes(
                query=search_text,
                limit=limit * 3 if use_reranking else limit,
                person_names=person_names if person_names else None,
                face_cluster_ids=face_ids if face_ids else None,
                clothing_colors=all_clothing_colors or None,
                clothing_types=all_clothing_types or None,
                accessories=all_accessories or None,
                location=parsed.location if hasattr(parsed, "location") else None,
                visible_text=parsed.text_to_find if hasattr(parsed, "text_to_find") else None,
                action_keywords=parsed.action_keywords if hasattr(parsed, "action_keywords") else None,
                video_path=video_path,
                mood=parsed.mood,
                shot_type=parsed.shot_type,
                aesthetic_score=parsed.aesthetic_score,
                search_mode="hybrid",
                exclusions=parsed.exclusions if hasattr(parsed, "exclusions") else None,
            )
            all_results["scenes"] = scene_results
            log(f"[SOTA] Scenes: {len(scene_results)} results")

            if not scene_results and (person_names or all_clothing_colors or parsed.location):
                log("[SOTA] Strict scene search yielded 0 results. Retrying relaxed...")
                try:
                    fallback_scenes = await self.db.search_scenes(
                        query=search_text, limit=limit, search_mode="hybrid",
                    )
                    log(f"[SOTA] Relaxed search found {len(fallback_scenes)} scenes")
                    existing_ids = {r["id"] for r in scene_results}
                    for r in fallback_scenes:
                        if r["id"] not in existing_ids:
                            r["match_type"] = "relaxed"
                            scene_results.append(r)
                    all_results["scenes"] = scene_results
                except Exception as e:
                    log(f"[SOTA] Relaxed search failed: {e}")
        except Exception as e:
            log(f"[SOTA] Scene search failed: {e}")
            all_results["scenes"] = []

        try:
            scenelet_results = await self.db.search_scenelets(
                query=search_text,
                limit=limit * 3 if use_reranking else limit,
                video_path=video_path,
            )
            all_results["scenelets"] = scenelet_results
            log(f"[SOTA] Scenelets: {len(scenelet_results)} results")
        except Exception as e:
            log(f"[SOTA] Scenelet search failed: {e}")
            all_results["scenelets"] = []

        try:
            frame_results = await self.db.search_frames_hybrid(
                query=search_text,
                limit=limit * 3 if use_reranking else limit,
                face_cluster_ids=face_ids if face_ids else None,
                video_paths=[video_path] if video_path else None,
            )
            all_results["frames"] = frame_results
            log(f"[SOTA] Frames: {len(frame_results)} results")
        except Exception as e:
            log(f"[SOTA] Frame search failed: {e}")
            all_results["frames"] = []

        try:
            voice_results = await self.db.search_voice_segments(
                query=search_text, limit=limit * 2, video_path=video_path,
            )
            all_results["voice"] = voice_results if voice_results else []
            log(f"[SOTA] Voice: {len(all_results['voice'])} results")
        except Exception as e:
            log(f"[SOTA] Voice search failed: {e}")
            all_results["voice"] = []

        try:
            audio_results = await self.db.search_audio_events_semantic(
                query=search_text, limit=limit * 2, video_path=video_path,
            )
            all_results["audio_events"] = audio_results if audio_results else []
            log(f"[SOTA] Audio events: {len(all_results['audio_events'])} results")
        except Exception as e:
            log(f"[SOTA] Audio event search failed: {e}")
            all_results["audio_events"] = []

        try:
            dialogue_results = await self.db.search_dialogue(
                query=search_text, limit=limit * 2, video_path=video_path,
            )
            all_results["dialogue"] = dialogue_results if dialogue_results else []
            log(f"[SOTA] Dialogue: {len(all_results['dialogue'])} results")
        except Exception as e:
            log(f"[SOTA] Dialogue search failed: {e}")
            all_results["dialogue"] = []

        try:
            video_meta = await self.db.search_video_metadata(query=search_text, limit=limit)
            all_results["video_metadata"] = video_meta if video_meta else []
            log(f"[SOTA] Video metadata: {len(all_results['video_metadata'])} results")
        except Exception as e:
            log(f"[SOTA] Video metadata search failed: {e}")
            all_results["video_metadata"] = []

        # Voice identity boost
        if person_names and all_results.get("voice"):
            for v_result in all_results["voice"]:
                speaker_name = str(v_result.get("speaker_name", ""))
                if any(name.lower() in speaker_name.lower() for name in person_names):
                    v_result["score"] = min(1.0, v_result.get("score", 0.5) * 1.5)
                    v_result["identity_boosted"] = True

        # 4. Adaptive weighting + RRF fusion
        weights = self._compute_adaptive_weights(search_text, parsed)
        fused_scores = defaultdict(float)
        result_data = {}
        k = settings.rrf_constant

        for modality, results in all_results.items():
            weight = weights.get(modality, 0.1)
            for rank, result in enumerate(results, start=1):
                result_id = result.get("id")
                if not result_id:
                    continue

                vp = result.get("video_path") or result.get("media_path") or ""
                start_time = (
                    result.get("start_time") or result.get("start")
                    or result.get("timestamp") or 0
                )

                bucket_size = settings.timestamp_bucket_seconds
                ts_float = float(start_time)
                primary_bucket = int(ts_float / bucket_size) * int(bucket_size)
                half_offset_bucket = int((ts_float + bucket_size / 2) / bucket_size) * int(bucket_size)

                fusion_keys = [f"{vp}:{primary_bucket}"]
                if half_offset_bucket != primary_bucket:
                    fusion_keys.append(f"{vp}:{half_offset_bucket}")

                rrf_score = 1.0 / (k + rank)
                weighted_score = rrf_score * weight

                for fusion_key in fusion_keys:
                    fused_scores[fusion_key] += weighted_score

                    if fusion_key not in result_data:
                        result_data[fusion_key] = result.copy()
                        result_data[fusion_key]["modality_sources"] = []
                        result_data[fusion_key]["_original_id"] = result_id
                        result_data[fusion_key]["_best_timestamp_score"] = weighted_score
                    else:
                        existing = result_data[fusion_key]
                        if weighted_score > existing.get("_best_timestamp_score", 0):
                            for ts_key in ["start_time", "end_time", "start", "end", "timestamp"]:
                                if result.get(ts_key) is not None:
                                    existing[ts_key] = result[ts_key]
                            existing["_best_timestamp_score"] = weighted_score

                    if modality not in result_data[fusion_key]["modality_sources"]:
                        result_data[fusion_key]["modality_sources"].append(modality)

                    existing = result_data[fusion_key]
                    if not existing.get("face_names") and result.get("face_names"):
                        existing["face_names"] = result["face_names"]
                    if not existing.get("person_names") and result.get("person_names"):
                        existing["person_names"] = result["person_names"]
                    if not existing.get("face_cluster_ids") and result.get("face_cluster_ids"):
                        existing["face_cluster_ids"] = result["face_cluster_ids"]
                    if not existing.get("description") and result.get("description"):
                        existing["description"] = result["description"]
                    if not existing.get("entities") and result.get("entities"):
                        existing["entities"] = result["entities"]
                    if not existing.get("scene_location") and result.get("scene_location"):
                        existing["scene_location"] = result["scene_location"]

        ranked = sorted(fused_scores.items(), key=lambda x: x[1], reverse=True)[
            : limit * 3 if use_reranking else limit
        ]

        candidates = []
        max_rrf_score = max(ranked[0][1], 1e-9) if ranked else 1.0

        for fusion_key, score in ranked:
            result = result_data[fusion_key]
            normalized_score = score / max_rrf_score
            result["score"] = normalized_score
            result["fused_score"] = score

            face_names = result.get("face_names") or []
            person_names_r = result.get("person_names") or []
            face_cluster_ids = result.get("face_cluster_ids") or []

            if not face_names and face_cluster_ids:
                resolved_names = []
                for cid in face_cluster_ids:
                    try:
                        name = self.db.get_face_name_by_cluster(cid)
                        if name:
                            resolved_names.append(name)
                    except Exception:
                        pass
                if resolved_names:
                    face_names = resolved_names

            if not face_names and person_names_r:
                face_names = person_names_r

            result["face_names"] = face_names

            sources = result.get("modality_sources", [])
            result["_debug"] = {
                "modalities_used": sources,
                "raw_fused_score": score,
                "normalized_score": normalized_score,
                "models_contributed": self._get_models_for_modalities(sources),
                "match_type": "multimodal_fusion" if len(sources) > 1 else "single_modality",
            }

            if sources:
                source_desc = ", ".join(sources)
                desc_preview = (
                    result.get("description") or result.get("action")
                    or result.get("visual_summary") or ""
                )
                reason_parts = []
                if face_names:
                    reason_parts.append(f"People: {', '.join(face_names)}")
                if result.get("scene_location"):
                    reason_parts.append(f"Location: {result['scene_location']}")
                if result.get("entities"):
                    entities = result["entities"][:5]
                    reason_parts.append(f"Objects: {', '.join(entities)}")

                if reason_parts:
                    reason_extra = " | ".join(reason_parts)
                    result["match_reason"] = f"Matched via {source_desc} ({normalized_score * 100:.0f}%) - {reason_extra}"
                elif desc_preview:
                    result["match_reason"] = f"Semantic match (score={normalized_score:.2f}): {desc_preview[:100]}..."
                else:
                    result["match_reason"] = f"Matched via {source_desc} (score={normalized_score:.2f})"

            candidates.append(result)

        log(f"[SOTA] After weighted fusion: {len(candidates)} candidates")

        fallback_used = None
        if not any(all_results.values()):
            fallback_used = "no_results"

        # 5. BGE Reranking
        if use_reranking and candidates:
            try:
                pairs = []
                for c in candidates:
                    desc = c.get("description") or c.get("visual_summary") or ""
                    action = c.get("motion_text") or c.get("action_summary") or ""
                    content = f"{desc} {action}"
                    if c.get("dialogue_transcript"):
                        content += f" Dialogue: {c['dialogue_transcript']}"
                    pairs.append([query, content])

                from sentence_transformers import CrossEncoder
                from core.processing.resource_arbiter import RESOURCE_ARBITER

                reranker_model_id = "BAAI/bge-reranker-v2-m3"

                def _cleanup_reranker():
                    if hasattr(self, "_reranker") and self._reranker is not None:
                        del self._reranker
                        self._reranker = None
                        try:
                            import torch
                            if torch.cuda.is_available():
                                torch.cuda.empty_cache()
                        except Exception:
                            pass

                await RESOURCE_ARBITER.acquire(
                    "bge_reranker", vram_gb=1.5, cleanup_fn=_cleanup_reranker
                )

                if not hasattr(self, "_reranker") or self._reranker is None:
                    self._reranker = CrossEncoder(reranker_model_id, trust_remote_code=True)

                scores = self._reranker.predict(pairs)

                for i, score in enumerate(scores):
                    candidates[i]["score"] = float(score)
                    candidates[i]["rerank_score"] = float(score)

                candidates.sort(key=lambda x: x["score"], reverse=True)
                candidates = candidates[:limit]
                log(f"[SOTA Search] BGE Reranked to {len(candidates)} results")

            except Exception as e:
                log.error(f"[SOTA Search] Reranking failed: {e}")

        # 6. Granular scoring
        has_constraints = any([
            getattr(parsed, "identities", None),
            getattr(parsed, "clothing", None),
            getattr(parsed, "text", None),
            getattr(parsed, "actions", None),
            getattr(parsed, "location", None),
            getattr(parsed, "audio", None),
            getattr(parsed, "spatial", None),
            getattr(parsed, "exclusions", None),
        ])
        if has_constraints:
            candidates = self._apply_granular_scoring(candidates, parsed)
            candidates.sort(key=lambda x: x.get("score", 0), reverse=True)

        pipeline_steps = [
            {"step": "Query Parsing", "status": "completed",
             "detail": f"Extracted {len(parsed.entities) if hasattr(parsed, 'entities') and parsed.entities else 0} entities",
             "data": {"original_query": query[:100]}},
            {"step": "Vector Search", "status": "completed",
             "detail": f"{'Fallback: ' + str(fallback_used) if fallback_used else 'scenes'} → {len(candidates)} results",
             "data": {"collection_searched": fallback_used or "scenes",
                      "fallback_used": fallback_used is not None,
                      "candidates_found": len(candidates)}},
            {"step": "LLM Reranking", "status": "completed" if use_reranking else "skipped",
             "detail": f"{'Applied' if use_reranking else 'Disabled'} → {len(candidates[:limit])} final results",
             "data": {"enabled": use_reranking, "final_count": len(candidates[:limit])}},
        ]

        reasoning_chain = {
            "step1_parse": f"Parsed '{query[:50]}...' → dynamic entities",
            "step2_identity": f"Resolved: {person_names} → {len(face_ids)} faces",
            "step3_expand": f"Search text: {search_text[:80]}...",
            "step4_retrieve": f"Retrieved {len(candidates)} candidates from {fallback_used or 'scenes'}",
            "step5_rerank": f"Reranking={use_reranking}, final={len(candidates[:limit])}",
        }

        return {
            "query": query,
            "parsed": parsed.model_dump() if hasattr(parsed, "model_dump") else {},
            "search_text": search_text,
            "person_names_resolved": person_names,
            "face_ids_matched": len(face_ids),
            "results": candidates[:limit],
            "result_count": len(candidates[:limit]),
            "search_type": "sota",
            "reranking_used": use_reranking,
            "fallback_used": fallback_used,
            "pipeline_steps": pipeline_steps,
            "reasoning_chain": reasoning_chain,
        }

    def _compute_adaptive_weights(self, query: str, parsed: Any) -> dict[str, float]:
        llm_weights = None
        if parsed and hasattr(parsed, "modality_weights") and parsed.modality_weights:
            llm_weights = parsed.modality_weights
        elif parsed and isinstance(parsed, dict) and "modality_weights" in parsed:
            llm_weights = parsed["modality_weights"]

        if llm_weights:
            weights = {
                "scenes": llm_weights.get("visual", 0.2),
                "frames": llm_weights.get("visual", 0.2) * 0.8,
                "scenelets": llm_weights.get("action", 0.15),
                "voice": llm_weights.get("identity", 0.15),
                "dialogue": llm_weights.get("dialogue", 0.2),
                "audio_events": llm_weights.get("audio", 0.1),
                "video_metadata": llm_weights.get("context", 0.05),
            }
            total = sum(weights.values())
            if total > 0:
                weights = {k: v / total for k, v in weights.items()}
            log(f"[ADAPTIVE] Using LLM-inferred weights: {weights}")
        else:
            weights = {k: 1.0 / 7.0 for k in [
                "scenes", "frames", "scenelets", "voice",
                "dialogue", "audio_events", "video_metadata",
            ]}
            log("[ADAPTIVE] Query weights: UNIFORM (no LLM decomposition)")

        return weights

    # =========================================================================
    # SCENELET + MULTIMODAL + TEMPORAL
    # =========================================================================

    @observe("scenelet_search")
    async def scenelet_search(
        self, query: str, video_path: str | None = None, limit: int = 10,
    ) -> dict:
        log(f"[Scenelet Search] Query: '{query[:80]}...'")
        parsed = await self.parse_query(query)
        search_text = parsed.to_search_text() or query

        results = self.db.search_scenelets(
            query=search_text, limit=limit, video_path=video_path,
            gap_threshold=3.0, padding=3.0,
        )

        formatted_results = []
        for r in results:
            start = r.get("start_time", 0.0)
            end = r.get("end_time", 0.0)
            text = r.get("text", "")
            formatted_results.append({
                **r,
                "reasoning": f"Matched scenelet ({start:.1f}s-{end:.1f}s): {text[:100]}...",
                "match_explanation": f"Action sequence detected: {text[:50]}",
            })

        return {
            "query": query, "search_type": "scenelet",
            "results": formatted_results, "result_count": len(formatted_results),
        }

    @observe("comprehensive_multimodal_search")
    async def comprehensive_multimodal_search(
        self,
        query: str,
        limit: int = 20,
        video_path: str | None = None,
        use_reranking: bool = True,
    ) -> dict[str, Any]:
        log(f"[Multimodal] Comprehensive search: '{query[:80]}...'")

        parsed = await self.parse_query(query)
        search_text = parsed.to_search_text() or query

        person_names = []
        face_cluster_ids = []
        voice_cluster_ids = []

        if hasattr(parsed, "entities") and parsed.entities:
            for entity in parsed.entities:
                if entity.entity_type.lower() == "person" and entity.name:
                    person_names.append(entity.name)
        elif parsed.person_name:
            person_names.append(parsed.person_name)

        for name in person_names:
            face_cid = self.db.get_face_cluster_by_name(name)
            if face_cid:
                face_cluster_ids.append(face_cid)
            voice_cid = self.db.get_speaker_cluster_by_name(name)
            if voice_cid:
                voice_cluster_ids.append(voice_cid)

        log(f"[Multimodal] Resolved identities: faces={face_cluster_ids}, voices={voice_cluster_ids}")

        modality_results = {}

        try:
            scene_results = await self.db.search_scenes(
                query=search_text, limit=limit * 2,
                person_name=person_names[0] if person_names else None,
                face_cluster_ids=face_cluster_ids if face_cluster_ids else None,
                clothing_color=getattr(parsed, "clothing_color", None),
                clothing_type=getattr(parsed, "clothing_type", None),
                location=getattr(parsed, "location", None),
                visible_text=getattr(parsed, "text_to_find", None),
                action_keywords=getattr(parsed, "action_keywords", None),
                video_path=video_path, search_mode="hybrid",
            )
            modality_results["scenes"] = scene_results
            log(f"[Multimodal] Scene search: {len(scene_results)} results")
        except Exception as e:
            log(f"[Multimodal] Scene search failed: {e}")
            modality_results["scenes"] = []

        try:
            if person_names or "speak" in query.lower() or "say" in query.lower():
                voice_results = []
                if video_path:
                    voice_segments = self.db.get_voice_segments_by_video(video_path=video_path)
                else:
                    voice_segments = self.db.get_all_voice_segments(limit=500)

                for seg in voice_segments:
                    speaker_name = seg.get("speaker_name", "")
                    if any(name.lower() in str(speaker_name).lower() for name in person_names):
                        voice_results.append({
                            "id": seg.get("id"),
                            "video_path": seg.get("media_path"),
                            "start_time": seg.get("start_time", seg.get("start", 0)),
                            "end_time": seg.get("end_time", seg.get("end", 0)),
                            "speaker_name": speaker_name,
                            "score": 0.9, "modality": "voice",
                        })
                modality_results["voices"] = voice_results[:limit]
                log(f"[Multimodal] Voice search: {len(voice_results)} matches")
        except Exception as e:
            log(f"[Multimodal] Voice search failed: {e}")
            modality_results["voices"] = []

        try:
            if len(person_names) >= 2 or "with" in query.lower() or "together" in query.lower():
                co_occurrences = self.db.get_person_co_occurrences(video_path=video_path)
                co_results = []
                for co in co_occurrences:
                    p1_name = co.get("person1_name", "")
                    p2_name = co.get("person2_name", "")
                    matched = any(
                        name.lower() in str(p1_name).lower() or name.lower() in str(p2_name).lower()
                        for name in person_names
                    )
                    if matched:
                        co_results.append({
                            "video_path": co.get("video_path"),
                            "start_time": co.get("start_time", 0),
                            "end_time": co.get("end_time", 0),
                            "person1": p1_name, "person2": p2_name,
                            "interaction_count": co.get("interaction_count", 1),
                            "score": min(1.0, 0.5 + co.get("interaction_count", 1) * 0.1),
                            "modality": "co_occurrence",
                        })
                modality_results["co_occurrences"] = co_results[:limit]
                log(f"[Multimodal] Co-occurrence search: {len(co_results)} relationships")
        except Exception as e:
            log(f"[Multimodal] Co-occurrence search failed: {e}")
            modality_results["co_occurrences"] = []

        try:
            audio_events = await self.db.search_audio_events(query=search_text, limit=limit)
            modality_results["audio_events"] = [
                {**event, "modality": "audio_event", "score": event.get("score", 0.7)}
                for event in audio_events
            ]
            log(f"[Multimodal] Audio events search: {len(audio_events)} matches")
        except Exception as e:
            log(f"[Multimodal] Audio events search failed: {e}")
            modality_results["audio_events"] = []

        try:
            video_meta = await self.db.search_video_metadata(query=search_text, limit=limit)
            modality_results["video_metadata"] = [
                {**meta, "modality": "video_metadata", "score": meta.get("score", 0.6)}
                for meta in video_meta
            ]
            log(f"[Multimodal] Video metadata search: {len(video_meta)} matches")
        except Exception as e:
            log(f"[Multimodal] Video metadata search failed: {e}")
            modality_results["video_metadata"] = []

        # RRF fusion
        fused_results = self._rrf_fusion_multimodal(modality_results, limit * 2)
        log(f"[Multimodal] RRF fusion: {len(fused_results)} combined results")

        # Reranking
        if use_reranking and fused_results:
            try:
                from core.retrieval.reranker import SearchCandidate
                sc_candidates = [
                    SearchCandidate(
                        video_path=str(r.get("video_path", "")),
                        start_time=float(r.get("start_time", 0)),
                        end_time=float(r.get("end_time", 0)),
                        score=float(r.get("fused_score", r.get("score", 0))),
                        payload=r,
                    )
                    for r in fused_results[:limit * 2]
                ]
                ranked = await self.council.council_rerank(
                    query=query, candidates=sc_candidates, max_candidates=limit, use_vlm=True,
                )
                fused_results = []
                for r in ranked:
                    result = r.candidate.payload.copy()
                    result["final_score"] = r.final_score
                    result["llm_reasoning"] = r.vlm_reason or "Verified"
                    fused_results.append(result)
                log(f"[Multimodal] Reranked to {len(fused_results)} final results")
            except Exception as e:
                log(f"[Multimodal] Reranking failed: {e}")

        return {
            "query": query,
            "search_type": "comprehensive_multimodal",
            "parsed": parsed.model_dump() if hasattr(parsed, "model_dump") else {},
            "identities_resolved": {
                "names": person_names,
                "face_clusters": face_cluster_ids,
                "voice_clusters": voice_cluster_ids,
            },
            "modality_breakdown": {
                "scenes_searched": len(modality_results.get("scenes", [])),
                "voices_matched": len(modality_results.get("voices", [])),
                "co_occurrences_found": len(modality_results.get("co_occurrences", [])),
                "audio_events_matched": len(modality_results.get("audio_events", [])),
                "video_metadata_matched": len(modality_results.get("video_metadata", [])),
            },
            "results": fused_results[:limit],
            "result_count": len(fused_results[:limit]),
            "all_modalities_used": True,
        }

    @observe("temporal_sequence_search")
    async def search_temporal_sequence(
        self,
        sequence_steps: list[dict[str, str]],
        video_path: str | None = None,
        max_gap_seconds: float = 60.0,
        limit: int = 10,
    ) -> dict[str, Any]:
        from core.storage.identity_graph import identity_graph

        if not sequence_steps or len(sequence_steps) < 2:
            return {"error": "Temporal sequence requires at least 2 steps", "results": []}

        media_ids: list[str] = []
        if video_path:
            media_ids = [video_path]
        else:
            try:
                import sqlite3
                with identity_graph._lock, sqlite3.connect(identity_graph.db_path) as conn:
                    cursor = conn.execute("SELECT DISTINCT media_id FROM scenes LIMIT 100")
                    media_ids = [row[0] for row in cursor.fetchall()]
            except Exception as e:
                log(f"[Temporal] Failed to get media list: {e}")
                return {"error": str(e), "results": []}

        all_chains: list[dict] = []

        for media_id in media_ids:
            scenes = identity_graph.get_scenes_for_media(media_id)
            if len(scenes) < len(sequence_steps):
                continue

            person_cluster_map: dict[str, list[int]] = {}
            for step in sequence_steps:
                person_name = step.get("person", "").strip()
                if person_name and person_name not in person_cluster_map:
                    try:
                        ids = await self.db.resolve_person_ids(person_name)
                        person_cluster_map[person_name] = ids or []
                    except Exception:
                        person_cluster_map[person_name] = []

            for start_idx in range(len(scenes) - len(sequence_steps) + 1):
                chain_matches = []
                chain_score = 0.0
                prev_end_time = None
                valid_chain = True

                for step_idx, step in enumerate(sequence_steps):
                    best_match = None
                    best_score = 0.0
                    search_start = start_idx + step_idx if step_idx == 0 else start_idx + len(chain_matches)

                    for scene_idx in range(search_start, min(search_start + 3, len(scenes))):
                        scene = scenes[scene_idx]
                        if prev_end_time is not None:
                            gap = scene.start_time - prev_end_time
                            if gap > max_gap_seconds:
                                continue

                        score = self._score_scene_step_match(scene, step, person_cluster_map)
                        if score > best_score:
                            best_score = score
                            best_match = scene

                    if best_match and best_score > 0.2:
                        chain_matches.append({
                            "step": step,
                            "scene_id": best_match.id,
                            "start_time": best_match.start_time,
                            "end_time": best_match.end_time,
                            "description": best_match.description,
                            "location": best_match.location,
                            "actions": best_match.actions,
                            "score": best_score,
                        })
                        chain_score += best_score
                        prev_end_time = best_match.end_time
                    else:
                        valid_chain = False
                        break

                if valid_chain and len(chain_matches) == len(sequence_steps):
                    all_chains.append({
                        "media_path": media_id,
                        "chain": chain_matches,
                        "total_score": chain_score / len(sequence_steps),
                        "time_span": {
                            "start": chain_matches[0]["start_time"],
                            "end": chain_matches[-1]["end_time"],
                            "duration": chain_matches[-1]["end_time"] - chain_matches[0]["start_time"],
                        },
                    })

        all_chains.sort(key=lambda x: x["total_score"], reverse=True)
        return {
            "query_steps": sequence_steps,
            "results": all_chains[:limit],
            "result_count": len(all_chains[:limit]),
            "total_found": len(all_chains),
        }
