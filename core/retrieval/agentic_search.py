"""Agentic Search with LLM Query Expansion and Scene-Level Search.

Uses LLM to expand queries intelligently at search time,
NOT hardcoded synonyms during ingestion. Prompts loaded from external files.
"""

from __future__ import annotations

import asyncio
from typing import TYPE_CHECKING, Any

from core.knowledge.schemas import ParsedQuery
from core.retrieval.reranker import RerankingCouncil
from core.utils.logger import log
from core.utils.observe import observe
from core.utils.prompt_loader import load_prompt
from llm.factory import LLMFactory

if TYPE_CHECKING:
    from core.storage.db import VectorDB
    from llm.interface import LLMInterface


# Load prompts from external files - NO HARDCODING
DYNAMIC_QUERY_PROMPT = load_prompt("dynamic_query")
QUERY_EXPANSION_PROMPT = DYNAMIC_QUERY_PROMPT  # Legacy alias


# Pydantic model for LLM reranking results (moved from loop to module level)
from pydantic import BaseModel, Field


class RerankResult(BaseModel):
    """Structured output for LLM-based reranking verification."""

    match_score: float = Field(default=0.5)
    constraints_checked: list[str] = Field(default_factory=list)
    reasoning: str = Field(default="")
    missing: list[str] = Field(default_factory=list)


class SearchAgent:
    """Agentic search with LLM-powered query expansion and scene-level search.

    Supports complex paragraph-length queries with multiple constraints:
    - Identity (face clusters)
    - Clothing color and type
    - Accessories
    - Location
    - Actions and outcomes
    - Visible text/brands
    """

    def __init__(
        self,
        db: VectorDB,
        llm: LLMInterface | None = None,
        enable_hybrid: bool = True,
    ) -> None:
        """Initializes the search agent.

        Args:
            db: Vector database for search operations.
            llm: Optional LLM interface for query expansion. If not provided,
                the default LLM from factory will be used.
            enable_hybrid: Enable hybrid BM25+vector search with RRF fusion.
        """
        self.db = db
        self.llm = llm or LLMFactory.create_llm()
        self._hybrid_searcher = None
        self._enable_hybrid = enable_hybrid
        self._council = None

        # Query embedding cache for instant repeat queries (OrderedDict for O(1) LRU)
        from collections import OrderedDict
        self._query_cache: OrderedDict[
            str, tuple[list[float], float]
        ] = OrderedDict()  # {hash: (vector, timestamp)}
        self._cache_ttl = 3600  # 1 hour TTL
        self._cache_max_size = 1000  # Max 1000 entries (~1MB)

    @property
    def hybrid_searcher(self):
        """Lazy-load HybridSearcher for BM25+vector fusion."""
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
        """Lazy-load RerankingCouncil."""
        if self._council is None:
            self._council = RerankingCouncil()
        return self._council

    async def _get_cached_embedding(self, query: str) -> list[float] | None:
        """Get cached query embedding if exists and not expired.

        Returns None if not cached or expired, requiring fresh encoding.
        """
        import hashlib
        import time

        cache_key = hashlib.md5(query.encode()).hexdigest()

        if cache_key in self._query_cache:
            vector, timestamp = self._query_cache[cache_key]
            if time.time() - timestamp < self._cache_ttl:
                # Move to end for LRU (most recently used)
                self._query_cache.move_to_end(cache_key)
                return vector
            else:
                # Expired, remove
                del self._query_cache[cache_key]

        return None

    def _cache_embedding(self, query: str, vector: list[float]) -> None:
        """Cache a query embedding with current timestamp."""
        import hashlib
        import time

        cache_key = hashlib.md5(query.encode()).hexdigest()

        # LRU eviction if cache is full - O(1) with OrderedDict
        if len(self._query_cache) >= self._cache_max_size:
            # Remove oldest (first) entry
            self._query_cache.popitem(last=False)

        self._query_cache[cache_key] = (vector, time.time())

    @observe("search_parse_query")
    async def parse_query(self, query: str) -> ParsedQuery:
        """Parses and expands a natural language search query using an LLM.

        Args:
            query: The user's natural language search query.

        Returns:
            A ParsedQuery object containing structured filters and expanded text.
        """
        prompt = QUERY_EXPANSION_PROMPT.format(query=query)

        try:
            # Timeout to prevent hanging if LLM is slow/stuck
            parsed = await asyncio.wait_for(
                self.llm.generate_structured(
                    schema=ParsedQuery,
                    prompt=prompt,
                    system_prompt="You are a search query parser. Return JSON only.",
                ),
                timeout=12.0,  # 12s massive timeout for slow machines, but finite
            )
            log(f"[Search] Parsed query: {parsed.model_dump()}")
            return parsed

        except Exception as e:
            log(f"[Search] Query parsing failed: {e}, using raw query")
            return ParsedQuery(visual_keywords=[query])

    def _resolve_identity(self, person_name: str) -> int | None:
        """Resolves a person's name to a face cluster ID.

        Performs exact and fuzzy matching against the HITL identity database.

        Args:
            person_name: The name of the person to resolve.

        Returns:
            The cluster ID if found, otherwise None.
        """
        try:
            # 1. Try exact match first (fastest)
            cluster_id = self.db.get_cluster_id_by_name(person_name)
            if cluster_id is not None:
                return cluster_id

            # 2. Fallback to fuzzy match (handles typos, partial names)
            cluster_id = self.db.fuzzy_get_cluster_id_by_name(person_name)
            if cluster_id:
                log(
                    f"[Search] Fuzzy matched '{person_name}' → cluster {cluster_id}"
                )
            return cluster_id
        except Exception:
            return None

    def _get_face_ids_for_cluster(self, cluster_id: int) -> list[str]:
        """Retrieves all face point IDs belonging to a specific cluster.

        Args:
            cluster_id: The ID of the face cluster.

        Returns:
            A list of face point IDs (strings).
        """
        try:
            return self.db.get_face_ids_by_cluster(cluster_id)
        except Exception:
            return []

    def _get_models_for_modalities(self, modalities: list[str]) -> list[str]:
        """Maps modality sources to the AI models that were used.

        Args:
            modalities: List of modality source names (scenes, frames, voice, etc.)

        Returns:
            List of model names that contributed to the search.
        """
        # Use config model map (Issue 3 fix)
        model_map = settings.modality_model_map
        models = set()
        for modality in modalities:
            models.update(model_map.get(modality, []))
        return list(models)

    @observe("search_scenes_agentic")
    async def search_scenes(
        self,
        query: str,
        limit: int = 20,
        use_expansion: bool = False,  # Default OFF to prevent LLM hallucination
        video_path: str | None = None,
    ) -> dict[str, Any]:
        """Performs a production-grade scene-level search with LLM expansion.

        Resolves identities, builds comprehensive filters for clothing,
        location, and actions, and queries the scene collection. Falls back
        to frame-level search if scene search fails.

        Args:
            query: The natural language search query.
            limit: Maximum number of results to return.
            use_expansion: Whether to use LLM query expansion.
            video_path: Optional filter for a specific video path.

        Returns:
            A dictionary containing results, parsed query, and metadata.
        """
        log(f"[Search] Scene search: '{query[:100]}...'")

        # 1. Parse and expand query
        if use_expansion:
            parsed = await self.parse_query(query)
        else:
            parsed = ParsedQuery(visual_keywords=[query])

        # 2. Resolve identity if person name found
        cluster_id: int | None = None
        face_ids: list[str] = []
        resolved_name: str | None = None

        if parsed.person_name:
            # 2a. Check Face Clusters (Automatic)
            cluster_id = self._resolve_identity(parsed.person_name)
            if cluster_id is not None:
                face_ids = self._get_face_ids_for_cluster(cluster_id)
                resolved_name = parsed.person_name
                log(f"[Search] Resolved '{parsed.person_name}' → cluster {cluster_id}")

            # 2b. Check Masklets (HITL / SAM 3) - The "Name Once" feature
            # Used for non-face objects or specific people tagged via SAM 3
            masklet_hits = await self.db.search_masklets(
                concept=parsed.person_name,
                limit=5
            )
            if masklet_hits:
                log(f"[Search] Found {len(masklet_hits)} tagged masklets for '{parsed.person_name}'")
                # Add found masklets directly to results
                for hit in masklet_hits:
                    # Convert masklet hit to result format
                    results.append({
                        "id": hit.get("id"),
                        "score": hit.get("score", 1.0) * 1.5, # Boost tracked objects
                        "type": "object_track",
                        "text": f"Tracked Object: {parsed.person_name}",
                        "start": hit.get("start_time"),
                        "end": hit.get("end_time"),
                        "video_path": hit.get("video_path"),
                        "thumbnail_url": f"/thumbnails/{hit.get('id')}.jpg" # Placeholder
                    })

        # 3. Build search query from expanded keywords
        search_text = parsed.to_search_text()
        log(f"[Search] Expanded search text: '{search_text}'")

        # 4. Execute scene search with comprehensive filters
        try:
            results = await self.db.search_scenes(
                query=search_text,
                limit=limit,
                person_name=resolved_name,
                face_cluster_ids=[cluster_id]
                if cluster_id is not None
                else None,
                clothing_color=parsed.clothing_color,
                clothing_type=parsed.clothing_type,
                accessories=parsed.accessories if parsed.accessories else None,
                location=parsed.location,
                visible_text=parsed.text_to_find
                if parsed.text_to_find
                else None,
                action_keywords=parsed.action_keywords
                if parsed.action_keywords
                else None,
                video_path=video_path,
                mood=parsed.mood,
                shot_type=parsed.shot_type,
                aesthetic_score=parsed.aesthetic_score,
                search_mode="hybrid",
            )
            log(f"[Search] Found {len(results)} scene results")
        except Exception as e:
            log(
                f"[Search] Scene search failed: {e}, falling back to frame search"
            )
            results = await self._fallback_frame_search(
                parsed, search_text, limit
            )

        # GOLD.md Compliance: Comprehensive Search Reasoning Trace
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
        log(
            f"[Search] Filters: faces={[cluster_id] if cluster_id else []}, "
            f"video={video_path or 'all'}"
        )
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
        use_expansion: bool = False,  # Default OFF to prevent LLM hallucination
    ) -> dict[str, Any]:
        """Performs a frame-level agentic search with query expansion.

        This is a legacy search method. For production use, use `search_scenes`.

        Args:
            query: The natural language search query.
            limit: Maximum number of results to return.
            use_expansion: Whether to use LLM query expansion.

        Returns:
            A dictionary containing results, parsed query, and metadata.
        """
        log(f"[Search] Agentic frame search: '{query}'")

        # 1. Parse and expand query
        if use_expansion:
            parsed = await self.parse_query(query)
        else:
            parsed = ParsedQuery(visual_keywords=[query])

        # 2. Resolve identity if person name found
        face_ids: list[str] = []
        resolved_name: str | None = None
        cluster_id: int | None = None

        if parsed.person_name:
            cluster_id = self._resolve_identity(parsed.person_name)
            if cluster_id is not None:
                face_ids = self._get_face_ids_for_cluster(cluster_id)
                resolved_name = parsed.person_name
                log(
                    f"[Search] Resolved '{parsed.person_name}' → cluster {cluster_id} ({len(face_ids)} faces)"
                )

        # 3. Build search query from expanded keywords
        search_text = parsed.to_search_text()
        log(f"[Search] Expanded search text: '{search_text}'")

        # 4. Build HYBRID FILTERS for precise matching
        from qdrant_client.http import models

        filters: list[models.FieldCondition] = []

        # Identity filter
        if cluster_id is not None:
            filters.append(
                models.FieldCondition(
                    key="face_cluster_ids",
                    match=models.MatchAny(any=[cluster_id]),
                )
            )
            log(f"[Search] Added identity filter: cluster_id={cluster_id}")

        # Clothing filters (new for complex queries)
        if parsed.clothing_color:
            filters.append(
                models.FieldCondition(
                    key="clothing_colors",
                    match=models.MatchAny(any=[parsed.clothing_color.lower()]),
                )
            )
            log(
                f"[Search] Added clothing color filter: {parsed.clothing_color}"
            )

        if parsed.clothing_type:
            filters.append(
                models.FieldCondition(
                    key="clothing_types",
                    match=models.MatchAny(any=[parsed.clothing_type.lower()]),
                )
            )
            log(f"[Search] Added clothing type filter: {parsed.clothing_type}")

        # Accessories filter
        if parsed.accessories:
            filters.append(
                models.FieldCondition(
                    key="accessories",
                    match=models.MatchAny(any=parsed.accessories),
                )
            )
            log(f"[Search] Added accessories filter: {parsed.accessories}")

        # Brand/Text filter
        if parsed.text_to_find:
            filters.append(
                models.FieldCondition(
                    key="visible_text",
                    match=models.MatchAny(any=parsed.text_to_find),
                )
            )
            log(f"[Search] Added text filter: {parsed.text_to_find}")

        # Location filter
        if parsed.location:
            filters.append(
                models.FieldCondition(
                    key="scene_location",
                    match=models.MatchText(text=parsed.location),
                )
            )
            log(f"[Search] Added location filter: {parsed.location}")

        # Entity/Object filter
        if parsed.visual_keywords:
            specific_entities = [
                k for k in parsed.visual_keywords if len(k) > 3
            ]
            if specific_entities:
                filters.append(
                    models.FieldCondition(
                        key="entity_names",
                        match=models.MatchAny(any=specific_entities),
                    )
                )
                log(f"[Search] Added entity filter: {specific_entities}")

        # 5. Execute HYBRID search (Vector + Filters)
        try:
            query_vector = (
                await self.db.encode_texts(
                    search_text or "scene activity", is_query=True
                )
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
                    {
                        "score": hit.score,
                        "id": str(hit.id),
                        **(hit.payload or {}),
                    }
                    for hit in results
                ]
            else:
                results = await self.db.search_frames(
                    query=search_text,
                    limit=limit,
                )
        except Exception as e:
            log(f"[Search] Hybrid search failed: {e}, falling back to simple")
            results = await self.db.search_frames(
                query=search_text,
                limit=limit,
            )

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
        self,
        parsed: ParsedQuery,
        search_text: str,
        limit: int,
    ) -> list[dict]:
        """Falls back to frame-level search if scene search is unavailable.

        Args:
            parsed: The already parsed query object.
            search_text: The expanded search text.
            limit: Maximum number of results to return.

        Returns:
            A list of search result dictionaries.
        """
        try:
            return await self.db.search_frames(query=search_text, limit=limit)
        except Exception:
            return []

    async def search_simple(self, query: str, limit: int = 20) -> list[dict]:
        """Performs a simple frame search without LLM expansion.

        Args:
            query: The search query string.
            limit: Maximum number of results to return.

        Returns:
            A list of search result dictionaries.
        """
        return await self.db.search_frames(query=query, limit=limit)

    async def hybrid_search(
        self,
        query: str,
        limit: int = 50,
        vector_weight: float = 0.7,
        keyword_weight: float = 0.3,
        video_id: str | None = None,
    ) -> list[dict]:
        """Perform hybrid BM25+vector search with weighted RRF fusion.

        Per AGENTS.MD: Use weighted RRF (0.7 vector, 0.3 BM25).

        Args:
            query: Search query string.
            limit: Maximum results to return.
            vector_weight: Weight for vector search (default 0.7).
            keyword_weight: Weight for BM25 keyword search (default 0.3).
            video_id: Optional filter by video.

        Returns:
            List of results with fused scores.
        """
        if self.hybrid_searcher:
            try:
                results = await self.hybrid_searcher.search(
                    query=query,
                    limit=limit,
                    vector_weight=vector_weight,
                    keyword_weight=keyword_weight,
                    video_id=video_id,
                )
                log(
                    f"[Search] Hybrid search: {len(results)} results "
                    f"(weights: {vector_weight:.1f}v/{keyword_weight:.1f}kw)"
                )
                return results
            except Exception as e:
                log(f"[Search] Hybrid search failed, falling back: {e}")

        # Fallback to vector-only search
        return await self.db.search_frames(query=query, limit=limit)

    # =========================================================================
    # SOTA SEARCH METHODS
    # =========================================================================

    @observe("search_rerank_llm")
    async def rerank_with_llm(
        self,
        query: str,
        candidates: list[dict],
        top_k: int = 10,
    ) -> list[dict]:
        """Re-ranks search candidates using an LLM to verify all query constraints.

        Performs a second-stage verification where an LLM checks if the
        candidate segments actually satisfy the specific entities, actions,
        and attributes mentioned in the query.

        Args:
            query: The original user search query.
            candidates: A list of candidate search results.
            top_k: The number of top results to return after re-ranking.

        Returns:
            The re-ranked list of results with LLM verification metadata.
        """
        if not candidates:
            return []

        # Load rerank prompt from external file
        rerank_prompt = load_prompt("rerank_verification")

        reranked = []

        for candidate in candidates[:top_k]:  # LLM rerank all top_k candidates
            description = (
                candidate.get("description", "")
                or candidate.get("dense_caption", "")
                or candidate.get("raw_description", "")
            )

            prompt = rerank_prompt.format(
                query=query,
                description=description[:500],
                face_names=candidate.get("face_names", []),
                location=candidate.get("location", "unknown"),
                actions=candidate.get("actions", []),
                visible_text=candidate.get("visible_text", []),
            )

            try:
                # Use module-level RerankResult class (moved from loop for performance)
                result = None
                for attempt in range(2):
                    try:
                        result = await self.llm.generate_structured(
                            schema=RerankResult,
                            prompt=prompt,
                            system_prompt="You are a video search result verifier. Return ONLY valid JSON with match_score, reasoning, constraints_checked, and missing fields.",
                        )
                        break
                    except Exception as retry_err:
                        if attempt == 0:
                            log(
                                f"[Rerank] Retry after parse error: {retry_err}"
                            )
                            continue
                        raise

                if result is None:
                    raise ValueError("LLM returned no result after retries")

                # Apply penalty for missing constraints
                missing_penalty = len(result.missing) * 0.1
                adjusted_score = max(0.0, result.match_score - missing_penalty)
                # Merge LLM verification with candidate
                candidate["llm_score"] = adjusted_score
                candidate["llm_reasoning"] = result.reasoning
                candidate["constraints_satisfied"] = result.constraints_checked
                candidate["constraints_missing"] = result.missing
                # Weight: configurable vector/LLM balance from settings
                from config import settings
                candidate["combined_score"] = (
                    candidate.get("score", 0) * settings.rerank_vector_weight
                    + adjusted_score * settings.rerank_llm_weight
                )
                reranked.append(candidate)

            except Exception as e:
                # If LLM fails, keep original score but add small penalty
                log(f"[Rerank] LLM verification failed: {e}")
                candidate["combined_score"] = candidate.get("score", 0) * 0.8
                candidate["llm_reasoning"] = (
                    f"Verification failed: {str(e)[:50]}"
                )
                reranked.append(candidate)

        # Sort by combined score
        reranked.sort(key=lambda x: x.get("combined_score", 0), reverse=True)
        return reranked[:top_k]

    @observe("search_sota")
    async def sota_search(
        self,
        query: str,
        limit: int = 10,
        video_path: str | None = None,
        use_reranking: bool = True,
        use_expansion: bool = False,  # Default OFF to prevent LLM hallucination
        expansion_fallback: bool = True,
    ) -> dict[str, Any]:
        """Performs a state-of-the-art search with full verification and explainability."""
        log(f"[SOTA Search] Query: '{query[:100]}...'")
        log(
            f"[SOTA Search] Options: expansion={use_expansion}, fallback={expansion_fallback}, rerank={use_reranking}"
        )

        # 1. Parse query with dynamic entity extraction (if enabled)
        expansion_used = False
        if use_expansion:
            try:
                parsed = await self.parse_query(query)
                search_text = parsed.to_search_text() or query
                expansion_used = search_text != query
                log(f"[SOTA Search] Expanded: '{search_text[:100]}...'")
            except Exception as e:
                log(f"[SOTA Search] Expansion failed: {e}, using raw query")
                parsed = ParsedQuery(visual_keywords=[query])
                search_text = query
        else:
            # Skip expansion - use raw query
            log("[SOTA Search] Expansion disabled, using raw query")
            parsed = ParsedQuery(visual_keywords=[query])
            search_text = query

        # 2. Resolve person identities
        person_names = []
        face_ids = []

        # Try new dynamic format first
        if hasattr(parsed, "entities") and parsed.entities:
            for entity in parsed.entities:
                if entity.entity_type.lower() == "person" and entity.name:
                    person_names.append(entity.name)
        # Fallback to legacy format
        elif parsed.person_name:
            person_names.append(parsed.person_name)

        for name in person_names:
            cluster_id = self._resolve_identity(name)
            if cluster_id:
                ids = self._get_face_ids_for_cluster(cluster_id)
                face_ids.extend(ids)
                log(f"[SOTA Search] Resolved '{name}' → {len(ids)} faces")

        all_results = {}

        try:
            scene_results = await self.db.search_scenes(
                query=search_text,
                limit=limit * 3 if use_reranking else limit,
                # FIX: Pass ALL person names, not just first
                person_names=person_names if person_names else None,
                face_cluster_ids=face_ids if face_ids else None,
                clothing_color=parsed.clothing_color
                if hasattr(parsed, "clothing_color")
                else None,
                clothing_type=parsed.clothing_type
                if hasattr(parsed, "clothing_type")
                else None,
                accessories=parsed.accessories
                if hasattr(parsed, "accessories")
                else None,
                location=parsed.location
                if hasattr(parsed, "location")
                else None,
                visible_text=parsed.text_to_find
                if hasattr(parsed, "text_to_find")
                else None,
                action_keywords=parsed.action_keywords
                if hasattr(parsed, "action_keywords")
                else None,
                video_path=video_path,
                mood=parsed.mood,
                shot_type=parsed.shot_type,
                aesthetic_score=parsed.aesthetic_score,
                search_mode="hybrid",
            )
            all_results["scenes"] = scene_results
            log(f"[SOTA] Scenes: {len(scene_results)} results")

            # CONSTRAINT RELAXATION: If strict search failed, try loose semantic search
            if not scene_results and (
                person_names or parsed.clothing_color or parsed.location
            ):
                log(
                    "[SOTA] Strict scene search yielded 0 results. Retrying with relaxed semantic search..."
                )
                try:
                    fallback_scenes = await self.db.search_scenes(
                        query=search_text,
                        limit=limit,
                        search_mode="hybrid",
                        # Intentionally omitting filters to cast a wider net
                    )
                    log(
                        f"[SOTA] Relaxed search found {len(fallback_scenes)} scenes"
                    )
                    # Append unique results (deduplicate by ID)
                    existing_ids = {r["id"] for r in scene_results}
                    for r in fallback_scenes:
                        if r["id"] not in existing_ids:
                            # Add a flag to indicate it satisfied loose constraints only
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
            # FIX: Use ALL face_ids, not just first one
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
                query=search_text,
                limit=limit * 2,
                video_path=video_path,
            )
            all_results["voice"] = voice_results if voice_results else []
            log(f"[SOTA] Voice: {len(all_results['voice'])} results")
        except Exception as e:
            log(f"[SOTA] Voice search failed: {e}")
            all_results["voice"] = []

        try:
            audio_results = await self.db.search_audio_events(
                query=search_text,
                limit=limit * 2,
            )
            all_results["audio_events"] = audio_results if audio_results else []
            log(
                f"[SOTA] Audio events: {len(all_results['audio_events'])} results"
            )
        except Exception as e:
            log(f"[SOTA] Audio event search failed: {e}")
            all_results["audio_events"] = []

        # DIALOGUE/TRANSCRIPT SEARCH - searches media_segments for spoken text/subs
        try:
            dialogue_results = await self.db.search_dialogue(
                query=search_text,
                limit=limit * 2,
                video_path=video_path,
            )
            all_results["dialogue"] = (
                dialogue_results if dialogue_results else []
            )
            log(f"[SOTA] Dialogue: {len(all_results['dialogue'])} results")
        except Exception as e:
            log(f"[SOTA] Dialogue search failed: {e}")
            all_results["dialogue"] = []

        from collections import defaultdict

        fused_scores = defaultdict(float)
        result_data = {}

        # QUERY-ADAPTIVE WEIGHTING: Detect query intent and adjust weights dynamically
        def _compute_adaptive_weights(
            query: str, parsed: Any
        ) -> dict[str, float]:
            """Compute modality weights based on detected query intent."""
            query_lower = query.lower()

            # Base weights from config (sum = 1.0)
            from config import settings
            weights = {
                "scenes": settings.modality_weight_scenes,
                "frames": settings.modality_weight_frames,
                "scenelets": settings.modality_weight_scenelets,
                "voice": settings.modality_weight_voice,
                "dialogue": settings.modality_weight_dialogue,
                "audio_events": settings.modality_weight_audio,
            }

            # Intent detection patterns
            visual_keywords = {
                "wearing",
                "shirt",
                "dress",
                "color",
                "blue",
                "red",
                "green",
                "looking",
                "standing",
                "sitting",
                "walking",
                "running",
                "temple",
                "near",
                "beside",
                "background",
                "scene",
                "shot",
                "frame",
                "image",
            }
            audio_keywords = {
                "music",
                "sound",
                "cheering",
                "clapping",
                "singing",
                "playing",
                "background music",
                "loud",
                "quiet",
                "noise",
                "crowd",
            }
            dialogue_keywords = {
                "said",
                "says",
                "saying",
                "spoke",
                "speaking",
                "told",
                "tells",
                "asked",
                "dialogue",
                "conversation",
                "talking",
                "speech",
                "words",
                "mentioned",
                "quote",
                "transcript",
            }
            identity_keywords = {
                "who",
                "person",
                "face",
                "speaker",
                "voice of",
                "recognize",
            }
            action_keywords = {
                "running",
                "jumping",
                "dancing",
                "fighting",
                "driving",
                "playing",
                "bowling",
                "kicking",
                "throwing",
                "catching",
                "hitting",
            }

            # INTENT CLASSIFICATION using keyword heuristics
            # Fast and reliable for common query patterns
            # NOTE: Semantic embedding-based classification was removed as it added latency
            # and required async calls which don't work in this sync context
            visual_score = sum(1 for kw in visual_keywords if kw in query_lower)
            audio_score = sum(1 for kw in audio_keywords if kw in query_lower)
            dialogue_score = sum(1 for kw in dialogue_keywords if kw in query_lower)
            identity_score = sum(1 for kw in identity_keywords if kw in query_lower)
            action_score = sum(1 for kw in action_keywords if kw in query_lower)


            # Also check parsed query for entities (visual) and person names (identity)
            if parsed:
                if hasattr(parsed, "entities") and parsed.entities:
                    visual_score += len(parsed.entities)
                if hasattr(parsed, "person_names") and parsed.person_names:
                    identity_score += (
                        len(parsed.person_names) * 2
                    )  # Strong signal

            # Determine dominant intent and adjust weights
            total_signal = (
                visual_score
                + audio_score
                + dialogue_score
                + identity_score
                + action_score
            )

            if total_signal > 0:
                detected_intents = []
                intent_profiles = []

                # Define weight profiles for each intent type
                VISUAL_PROFILE = {
                    "scenes": 0.30,
                    "frames": 0.25,
                    "scenelets": 0.15,
                    "voice": 0.10,
                    "dialogue": 0.10,
                    "audio_events": 0.10,
                }
                DIALOGUE_PROFILE = {
                    "scenes": 0.15,
                    "frames": 0.10,
                    "scenelets": 0.10,
                    "voice": 0.25,
                    "dialogue": 0.35,
                    "audio_events": 0.05,
                }
                AUDIO_PROFILE = {
                    "scenes": 0.15,
                    "frames": 0.10,
                    "scenelets": 0.20,
                    "voice": 0.05,
                    "dialogue": 0.15,
                    "audio_events": 0.35,
                }
                IDENTITY_PROFILE = {
                    "scenes": 0.25,
                    "frames": 0.20,
                    "scenelets": 0.10,
                    "voice": 0.25,
                    "dialogue": 0.15,
                    "audio_events": 0.05,
                }
                ACTION_PROFILE = {
                    "scenes": 0.25,
                    "frames": 0.20,
                    "scenelets": 0.30,
                    "voice": 0.10,
                    "dialogue": 0.10,
                    "audio_events": 0.05,
                }
                BALANCED_PROFILE = {
                    "scenes": 0.18,
                    "frames": 0.17,
                    "scenelets": 0.16,
                    "voice": 0.16,
                    "dialogue": 0.17,
                    "audio_events": 0.16,
                }

                # VISUAL-heavy query (clothing, objects, scene description)
                if visual_score >= 1:
                    intent_profiles.append((VISUAL_PROFILE, visual_score))
                    detected_intents.append("VISUAL")

                # DIALOGUE-heavy query (what someone said)
                if dialogue_score >= 1 or any(
                    kw in query_lower
                    for kw in ["said", "says", "telling", "speaking"]
                ):
                    intent_profiles.append(
                        (DIALOGUE_PROFILE, max(dialogue_score, 1))
                    )
                    detected_intents.append("DIALOGUE")

                # AUDIO-heavy query (sounds, music)
                if audio_score >= 1 or any(
                    kw in query_lower for kw in ["music", "sound", "cheering"]
                ):
                    intent_profiles.append((AUDIO_PROFILE, max(audio_score, 1)))
                    detected_intents.append("AUDIO")

                # IDENTITY-heavy query (person search)
                if identity_score >= 1 or (
                    parsed
                    and hasattr(parsed, "person_names")
                    and parsed.person_names
                ):
                    intent_profiles.append(
                        (IDENTITY_PROFILE, max(identity_score, 1))
                    )
                    detected_intents.append("IDENTITY")

                # ACTION-heavy query (motion, activity)
                if action_score >= 1:
                    intent_profiles.append((ACTION_PROFILE, action_score))
                    detected_intents.append("ACTION")

                # BLEND WEIGHTS based on detected intents
                if len(intent_profiles) == 0:
                    # No clear intent - use balanced weights
                    weights = BALANCED_PROFILE.copy()
                    detected_intents = ["BALANCED"]
                elif len(intent_profiles) == 1:
                    # Single intent - use that profile directly
                    weights = intent_profiles[0][0].copy()
                else:
                    # MULTIPLE INTENTS - blend weights proportionally
                    total_strength = sum(
                        strength for _, strength in intent_profiles
                    )
                    blended = {
                        "scenes": 0,
                        "frames": 0,
                        "scenelets": 0,
                        "voice": 0,
                        "dialogue": 0,
                        "audio_events": 0,
                    }

                    for profile, strength in intent_profiles:
                        weight_factor = strength / total_strength
                        for key in blended:
                            blended[key] += profile[key] * weight_factor

                    weights = blended
                    detected_intents.append(f"BLENDED({len(intent_profiles)})")

                log(
                    f"[ADAPTIVE] Query intent: {', '.join(detected_intents)} | Weights: scenes={weights['scenes']:.0%}, frames={weights['frames']:.0%}, dialogue={weights['dialogue']:.0%}, voice={weights['voice']:.0%}, audio={weights['audio_events']:.0%}"
                )

            return weights

        weights = _compute_adaptive_weights(search_text, parsed)

        # Use configurable RRF k from settings
        from config import settings
        k = settings.rrf_constant

        for modality, results in all_results.items():
            weight = weights.get(modality, 0.1)

            for rank, result in enumerate(results, start=1):
                result_id = result.get("id")
                if not result_id:
                    continue

                # Create a FUSION KEY based on video + timestamp for proper multi-modal merging
                # This allows same moment from different modalities to combine
                video_path = (
                    result.get("video_path") or result.get("media_path") or ""
                )
                start_time = (
                    result.get("start_time")
                    or result.get("start")
                    or result.get("timestamp")
                    or 0
                )

                # Use configurable timestamp bucket from settings
                time_bucket = int(float(start_time) / settings.timestamp_bucket_seconds) * int(settings.timestamp_bucket_seconds)
                fusion_key = f"{video_path}:{time_bucket}"

                rrf_score = 1.0 / (k + rank)
                weighted_score = rrf_score * weight

                fused_scores[fusion_key] += weighted_score

                # Keep best result for this fusion key, but track ALL modalities
                if fusion_key not in result_data:
                    result_data[fusion_key] = result.copy()
                    result_data[fusion_key]["modality_sources"] = []
                    result_data[fusion_key]["_original_id"] = result_id

                # Always add modality source (might be same moment from multiple modalities)
                if modality not in result_data[fusion_key]["modality_sources"]:
                    result_data[fusion_key]["modality_sources"].append(modality)

                # Merge richer data from other modalities
                existing = result_data[fusion_key]
                if not existing.get("face_names") and result.get("face_names"):
                    existing["face_names"] = result["face_names"]
                if not existing.get("person_names") and result.get(
                    "person_names"
                ):
                    existing["person_names"] = result["person_names"]
                if not existing.get("face_cluster_ids") and result.get(
                    "face_cluster_ids"
                ):
                    existing["face_cluster_ids"] = result["face_cluster_ids"]
                if not existing.get("description") and result.get(
                    "description"
                ):
                    existing["description"] = result["description"]
                if not existing.get("entities") and result.get("entities"):
                    existing["entities"] = result["entities"]
                if not existing.get("scene_location") and result.get(
                    "scene_location"
                ):
                    existing["scene_location"] = result["scene_location"]

        ranked = sorted(fused_scores.items(), key=lambda x: x[1], reverse=True)[
            : limit * 3 if use_reranking else limit
        ]

        candidates = []

        # FIXED: Use Linear Max-Scaling instead of Sigmoid
        # Sigmoid compresses RRF scores (which are small, ~0.01) into a narrow 0.5-0.6 range.
        # Linear scaling ensures the top result is 1.0 and preserves relative differences.
        max_rrf_score = max(ranked[0][1], 1e-9) if ranked else 1.0

        for fusion_key, score in ranked:
            result = result_data[fusion_key]

            # Linear Normalization: Top result = 1.0
            normalized_score = score / max_rrf_score
            result["score"] = normalized_score
            result["fused_score"] = score

            # ENSURE face_names is always present - resolve from face_cluster_ids if needed
            face_names = result.get("face_names") or []
            person_names = result.get("person_names") or []
            face_cluster_ids = result.get("face_cluster_ids") or []

            # If no face_names but we have cluster_ids, resolve them from DB
            if not face_names and face_cluster_ids:
                resolved_names = []
                for cluster_id in face_cluster_ids:
                    try:
                        name = self.db.get_face_name_by_cluster(cluster_id)
                        if name:
                            resolved_names.append(name)
                    except Exception:
                        pass
                if resolved_names:
                    face_names = resolved_names

            # Merge person_names if still empty
            if not face_names and person_names:
                face_names = person_names

            result["face_names"] = face_names

            # ADD DEBUG INFO for descriptive reasoning
            sources = result.get("modality_sources", [])
            result["_debug"] = {
                "modalities_used": sources,
                "raw_fused_score": score,
                "normalized_score": normalized_score,
                "models_contributed": self._get_models_for_modalities(sources),
                "match_type": "multimodal_fusion"
                if len(sources) > 1
                else "single_modality",
            }

            # Build descriptive match reason with actual content
            if sources:
                source_desc = ", ".join(sources)
                desc_preview = (
                    result.get("description")
                    or result.get("action")
                    or result.get("visual_summary")
                    or ""
                )

                # Build a richer reason with people and location
                reason_parts = []
                if face_names:
                    reason_parts.append(f"People: {', '.join(face_names)}")
                if result.get("scene_location"):
                    reason_parts.append(f"Location: {result['scene_location']}")
                if result.get("entities"):
                    entities = result["entities"][:5]  # First 5 entities
                    reason_parts.append(f"Objects: {', '.join(entities)}")

                if reason_parts:
                    reason_extra = " | ".join(reason_parts)
                    result["match_reason"] = (
                        f"Matched via {source_desc} ({normalized_score * 100:.0f}%) - {reason_extra}"
                    )
                elif desc_preview:
                    result["match_reason"] = (
                        f"Semantic match (score={normalized_score:.2f}): {desc_preview[:100]}..."
                    )
                else:
                    result["match_reason"] = (
                        f"Matched via {source_desc} (score={normalized_score:.2f})"
                    )

            candidates.append(result)

        log(f"[SOTA] After weighted fusion: {len(candidates)} candidates")

        fallback_used = None
        if not any(all_results.values()):
            fallback_used = "no_results"

        # 4. Re-rank with BGE-Reranker-v2-m3 (Single SOTA Reranker)
        if use_reranking and candidates:
            try:
                # Prepare pairs for reranking: [query, document_text]
                pairs = []
                for c in candidates:
                    # Construct rich representation for reranker
                    desc = c.get("description") or c.get("visual_summary") or ""
                    action = (
                        c.get("motion_text") or c.get("action_summary") or ""
                    )
                    content = f"{desc} {action}"
                    if c.get("dialogue_transcript"):
                        content += f" Dialogue: {c['dialogue_transcript']}"
                    pairs.append([query, content])

                # Use CrossEncoder or FlagReranker (Lazy Load)
                from sentence_transformers import CrossEncoder

                # We use a lightweight but powerful reranker model
                reranker_model_id = "BAAI/bge-reranker-v2-m3"
                # Note: In production we might want to singleton this.
                # For now, we assume it's cached or fast enough to load,
                # or we should use a shared instance from a factory.
                # To avoid re-loading every request, we should ideally put this in __init__
                # But to follow "the fix", we'll use a cached accessor or singleton if available.
                # Here we'll instantiate for correctness of logic.

                # Optimization: Check if self has reranker
                if not hasattr(self, "_reranker") or self._reranker is None:
                    self._reranker = CrossEncoder(
                        reranker_model_id, trust_remote_code=True
                    )

                scores = self._reranker.predict(pairs)

                # Update scores
                for i, score in enumerate(scores):
                    candidates[i]["score"] = float(score)  # Update main score
                    candidates[i]["rerank_score"] = float(score)

                # Re-sort
                candidates.sort(key=lambda x: x["score"], reverse=True)
                candidates = candidates[:limit]

                log(f"[SOTA Search] BGE Reranked to {len(candidates)} results")

            except Exception as e:
                log.error(f"[SOTA Search] Reranking failed: {e}")
                # Fallback to original order

        # 5. Granular Constraint Filtering & Scoring (if ParsedQuery has detail)
        # Only apply if we have granular constraints, to refine the final list
        if hasattr(parsed, "identities") and (
            parsed.identities or parsed.clothing or parsed.text
        ):
            candidates = self._apply_granular_scoring(candidates, parsed)
            # Re-sort after granular scoring adjustment
            candidates.sort(key=lambda x: x.get("score", 0), reverse=True)

        # GOLD.md Compliance: Comprehensive Search Reasoning Trace with Pipeline Steps
        # Build detailed pipeline_steps for frontend debug panel
        pipeline_steps = [
            {
                "step": "Query Parsing",
                "status": "completed",
                "detail": f"Extracted {len(parsed.entities) if hasattr(parsed, 'entities') and parsed.entities else 0} entities",
                "data": {"original_query": query[:100]},
            },
            {
                "step": "Vector Search",
                "status": "completed",
                "detail": f"{'Fallback: ' + str(fallback_used) if fallback_used else 'scenes'} → {len(candidates)} results",
                "data": {
                    "collection_searched": fallback_used or "scenes",
                    "fallback_used": fallback_used is not None,
                    "candidates_found": len(candidates),
                },
            },
            {
                "step": "LLM Reranking",
                "status": "completed" if use_reranking else "skipped",
                "detail": f"{'Applied' if use_reranking else 'Disabled'} → {len(candidates[:limit])} final results",
                "data": {
                    "enabled": use_reranking,
                    "final_count": len(candidates[:limit]),
                },
            },
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
            "parsed": parsed.model_dump()
            if hasattr(parsed, "model_dump")
            else {},
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

    def _apply_granular_scoring(
        self, results: list[dict], parsed: ParsedQuery
    ) -> list[dict]:
        """Refine scores based on granular constraint matching."""
        for result in results:
            base_score = float(result.get("score", 0.5))
            # Fetch payload (handle if result is dict or SearchCandidate)
            payload = (
                result.get("base_payload", result)
                if isinstance(result, dict)
                else result
            )

            # Text/Description for matching
            desc = (
                str(payload.get("description", ""))
                + " "
                + str(payload.get("action", ""))
            ).lower()
            ocr = (
                str(payload.get("ocr_text", ""))
                + " "
                + str(payload.get("visible_text", ""))
            ).lower()

            matches = 0
            total_checks = 0

            # Check Text Constraints
            if hasattr(parsed, "text"):
                for t in parsed.text:
                    total_checks += 1
                    if t.get("text", "").lower() in ocr:
                        matches += 1
                    elif t.get("text", "").lower() in desc:
                        matches += 0.5

            # Check Clothing Constraints
            if hasattr(parsed, "clothing"):
                for c in parsed.clothing:
                    total_checks += 1
                    c_item = c.get("item", "").lower()
                    c_color = c.get("color", "").lower()
                    if c_item and c_item in desc:
                        matches += 0.5
                    if c_color and c_color in desc:
                        matches += 0.5

            # Boost score if constraints match
            if total_checks > 0:
                boost = (matches / total_checks) * 0.2  # Max 20% boost
                if isinstance(result, dict):
                    result["score"] = base_score + boost
                    result["granular_matches"] = matches

        return results

    @observe("scenelet_search")
    async def scenelet_search(
        self,
        query: str,
        video_path: str | None = None,
        limit: int = 10,
    ) -> dict:
        """Searches for action-based video segments with temporal context.

        Identifies relevant moments and expands them into short windows
        (scenelets) to capture the full context of an action or event.

        Args:
            query: The natural language search query.
            video_path: Optional filter for a specific video.
            limit: Maximum number of results to return.

        Returns:
            A dictionary with results containing start and end time ranges.
        """
        log(f"[Scenelet Search] Query: '{query[:80]}...'")

        parsed = await self.parse_query(query)
        search_text = parsed.to_search_text() or query

        # Use dedicated Scenelet search from DB with aggressive clustering
        # gap_threshold=3.0 ensures we bridge 3s gaps (fixing "one part" issue)
        # padding=3.0 ensures we add context before/after (fixing "didnt pad" issue)
        results = self.db.search_scenelets(
            query=search_text,
            limit=limit,
            video_path=video_path,
            gap_threshold=3.0,
            padding=3.0,
        )

        formatted_results = []
        for r in results:
            # Format for API consistency
            start = r.get("start_time", 0.0)
            end = r.get("end_time", 0.0)
            text = r.get("text", "")
            score = r.get("score", 0.0)

            formatted_results.append(
                {
                    **r,
                    "reasoning": f"Matched scenelet ({start:.1f}s-{end:.1f}s): {text[:100]}...",
                    "match_explanation": f"Action sequence detected: {text[:50]}",
                }
            )

        return {
            "query": query,
            "search_type": "scenelet",
            "results": formatted_results,
            "result_count": len(formatted_results),
        }

    def _normalize_score(self, score: float) -> float:
        """Normalize score to 0-1 range to prevent negative values."""
        # Simple sigmoid-like or min-max if we knew bounds.
        # Since Cosine can be -1 to 1, or Dot Product can be large/negative.
        # We use a logistic function to squash output to 0-1 smoothly.
        import math

        try:
            # If score is very large negative, it goes to 0
            # If large positive, goes to 1
            # Scale factor 2.0 to make slope reasonable
            return 1 / (1 + math.exp(-2.0 * score))
        except OverflowError:
            return 0.0 if score < 0 else 1.0

    @observe("comprehensive_multimodal_search")
    async def comprehensive_multimodal_search(
        self,
        query: str,
        limit: int = 20,
        video_path: str | None = None,
        use_reranking: bool = True,
    ) -> dict[str, Any]:
        """Comprehensive search using ALL indexed data sources for maximum accuracy.

        This method combines data from ALL Qdrant collections:
        - scenes (visual/motion/dialogue vectors)
        - voice_segments (speaker diarization)
        - faces (identity clusters)
        - co-occurrences (temporal relationships)

        Uses Reciprocal Rank Fusion (RRF) to merge results from multiple modalities.

        Args:
            query: Natural language search query.
            limit: Maximum results to return.
            video_path: Optional filter to specific video.
            use_reranking: Whether to use LLM reranking.

        Returns:
            Dict with fused results and detailed modality breakdowns.
        """
        log(f"[Multimodal] Comprehensive search: '{query[:80]}...'")

        # === STEP 1: PARSE QUERY ===
        parsed = await self.parse_query(query)
        search_text = parsed.to_search_text() or query

        # === STEP 2: RESOLVE IDENTITIES (Face + Voice) ===
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
            # Face cluster
            face_cid = self.db.get_face_cluster_by_name(name)
            if face_cid:
                face_cluster_ids.append(face_cid)

            # Voice cluster (cross-modal linking)
            voice_cid = self.db.get_speaker_cluster_by_name(name)
            if voice_cid:
                voice_cluster_ids.append(voice_cid)

        log(
            f"[Multimodal] Resolved identities: faces={face_cluster_ids}, voices={voice_cluster_ids}"
        )

        # === STEP 3: SEARCH ALL MODALITIES ===
        modality_results = {}

        # 3a. Scene-level search (visual + motion + dialogue)
        try:
            scene_results = await self.db.search_scenes(
                query=search_text,
                limit=limit * 2,
                person_name=person_names[0] if person_names else None,
                face_cluster_ids=face_cluster_ids if face_cluster_ids else None,
                clothing_color=getattr(parsed, "clothing_color", None),
                clothing_type=getattr(parsed, "clothing_type", None),
                location=getattr(parsed, "location", None),
                visible_text=getattr(parsed, "text_to_find", None),
                action_keywords=getattr(parsed, "action_keywords", None),
                video_path=video_path,
                search_mode="hybrid",
            )
            modality_results["scenes"] = scene_results
            log(f"[Multimodal] Scene search: {len(scene_results)} results")
        except Exception as e:
            log(f"[Multimodal] Scene search failed: {e}")
            modality_results["scenes"] = []

        # 3b. Voice segment matching (for speaker queries)
        try:
            if (
                person_names
                or "speak" in query.lower()
                or "say" in query.lower()
            ):
                voice_results = []
                if video_path:
                    voice_segments = self.db.get_voice_segments_by_video(
                        video_path=video_path
                    )
                else:
                    voice_segments = self.db.get_all_voice_segments(limit=500)

                # Filter by speaker name if applicable
                for seg in voice_segments:
                    speaker_name = seg.get("speaker_name", "")
                    if any(
                        name.lower() in str(speaker_name).lower()
                        for name in person_names
                    ):
                        voice_results.append(
                            {
                                "id": seg.get("id"),
                                "video_path": seg.get("media_path"),
                                "start_time": seg.get(
                                    "start_time", seg.get("start", 0)
                                ),
                                "end_time": seg.get(
                                    "end_time", seg.get("end", 0)
                                ),
                                "speaker_name": speaker_name,
                                "score": 0.9,  # High score for exact name match
                                "modality": "voice",
                            }
                        )
                modality_results["voices"] = voice_results[:limit]
                log(f"[Multimodal] Voice search: {len(voice_results)} matches")
        except Exception as e:
            log(f"[Multimodal] Voice search failed: {e}")
            modality_results["voices"] = []

        # 3c. Co-occurrence relationships (for "X with Y" queries)
        try:
            if (
                len(person_names) >= 2
                or "with" in query.lower()
                or "together" in query.lower()
            ):
                co_occurrences = self.db.get_person_co_occurrences(
                    video_path=video_path
                )
                # Boost results where both mentioned people appear together
                co_results = []
                for co in co_occurrences:
                    p1_name = co.get("person1_name", "")
                    p2_name = co.get("person2_name", "")
                    # Check if any queried person appears
                    matched = False
                    for name in person_names:
                        if (
                            name.lower() in str(p1_name).lower()
                            or name.lower() in str(p2_name).lower()
                        ):
                            matched = True
                            break
                    if matched:
                        co_results.append(
                            {
                                "video_path": co.get("video_path"),
                                "start_time": co.get("start_time", 0),
                                "end_time": co.get("end_time", 0),
                                "person1": p1_name,
                                "person2": p2_name,
                                "interaction_count": co.get(
                                    "interaction_count", 1
                                ),
                                "score": min(
                                    1.0,
                                    0.5 + co.get("interaction_count", 1) * 0.1,
                                ),
                                "modality": "co_occurrence",
                            }
                        )
                modality_results["co_occurrences"] = co_results[:limit]
                log(
                    f"[Multimodal] Co-occurrence search: {len(co_results)} relationships"
                )
        except Exception as e:
            log(f"[Multimodal] Co-occurrence search failed: {e}")
            modality_results["co_occurrences"] = []

        # 3d. Audio events search (CLAP-detected sounds, music sections)
        try:
            audio_events = await self.db.search_audio_events(
                query=search_text,
                limit=limit,
            )
            modality_results["audio_events"] = [
                {
                    **event,
                    "modality": "audio_event",
                    "score": event.get("score", 0.7),
                }
                for event in audio_events
            ]
            log(
                f"[Multimodal] Audio events search: {len(audio_events)} matches"
            )
        except Exception as e:
            log(f"[Multimodal] Audio events search failed: {e}")
            modality_results["audio_events"] = []

        # 3e. Video metadata search (summaries, titles, context)
        try:
            video_meta = await self.db.search_video_metadata(
                query=search_text,
                limit=limit,
            )
            modality_results["video_metadata"] = [
                {
                    **meta,
                    "modality": "video_metadata",
                    "score": meta.get("score", 0.6),
                }
                for meta in video_meta
            ]
            log(
                f"[Multimodal] Video metadata search: {len(video_meta)} matches"
            )
        except Exception as e:
            log(f"[Multimodal] Video metadata search failed: {e}")
            modality_results["video_metadata"] = []

        # === STEP 4: RRF FUSION ACROSS MODALITIES ===
        fused_results = self._rrf_fusion_multimodal(modality_results, limit * 2)
        log(f"[Multimodal] RRF fusion: {len(fused_results)} combined results")

        # === STEP 5: LLM RERANKING (Optional) ===
        if use_reranking and fused_results:
            try:
                from core.retrieval.reranker import SearchCandidate

                sc_candidates = []
                for r in fused_results[: limit * 2]:
                    sc_candidates.append(
                        SearchCandidate(
                            video_path=str(r.get("video_path", "")),
                            start_time=float(r.get("start_time", 0)),
                            end_time=float(r.get("end_time", 0)),
                            score=float(
                                r.get("fused_score", r.get("score", 0))
                            ),
                            payload=r,
                        )
                    )

                ranked = await self.council.council_rerank(
                    query=query,
                    candidates=sc_candidates,
                    max_candidates=limit,
                    use_vlm=True,
                )

                fused_results = []
                for r in ranked:
                    result = r.candidate.payload.copy()
                    result["final_score"] = r.final_score
                    result["llm_reasoning"] = r.vlm_reason or "Verified"
                    fused_results.append(result)

                log(
                    f"[Multimodal] Reranked to {len(fused_results)} final results"
                )
            except Exception as e:
                log(f"[Multimodal] Reranking failed: {e}")

        # === STEP 6: BUILD RESPONSE ===
        return {
            "query": query,
            "search_type": "comprehensive_multimodal",
            "parsed": parsed.model_dump()
            if hasattr(parsed, "model_dump")
            else {},
            "identities_resolved": {
                "names": person_names,
                "face_clusters": face_cluster_ids,
                "voice_clusters": voice_cluster_ids,
            },
            "modality_breakdown": {
                "scenes_searched": len(modality_results.get("scenes", [])),
                "voices_matched": len(modality_results.get("voices", [])),
                "co_occurrences_found": len(
                    modality_results.get("co_occurrences", [])
                ),
                "audio_events_matched": len(
                    modality_results.get("audio_events", [])
                ),
                "video_metadata_matched": len(
                    modality_results.get("video_metadata", [])
                ),
            },
            "results": fused_results[:limit],
            "result_count": len(fused_results[:limit]),
            "all_modalities_used": True,
        }

    def _rrf_fusion_multimodal(
        self,
        modality_results: dict[str, list[dict]],
        limit: int = 50,
        k: int = 60,
    ) -> list[dict]:
        """Fuse results from multiple modalities using Reciprocal Rank Fusion.

        Args:
            modality_results: Dict mapping modality name to list of results.
            limit: Maximum results to return.
            k: RRF constant (default 60).

        Returns:
            Fused and sorted list of results.
        """
        # Build a unified score map keyed by (video_path, start_time rounded)
        score_map: dict[tuple, dict] = {}

        for modality, results in modality_results.items():
            if not results:
                continue

            for rank, result in enumerate(results):
                vp = result.get("video_path") or result.get("media_path") or ""
                st = round(
                    float(result.get("start_time", result.get("timestamp", 0))),
                    1,
                )
                key = (vp, st)

                # Calculate RRF score contribution
                rrf_score = 1.0 / (k + rank + 1)

                if key not in score_map:
                    score_map[key] = {
                        "video_path": vp,
                        "start_time": st,
                        "end_time": result.get("end_time", st + 5),
                        "fused_score": 0.0,
                        "modalities": [],
                        "description": result.get("description", ""),
                        "face_names": result.get("face_names", []),
                        "speaker_name": result.get("speaker_name"),
                    }

                score_map[key]["fused_score"] += rrf_score
                score_map[key]["modalities"].append(modality)

                # Merge additional data
                if (
                    result.get("description")
                    and not score_map[key]["description"]
                ):
                    score_map[key]["description"] = result["description"]
                if result.get("face_names"):
                    score_map[key]["face_names"] = list(
                        set(
                            score_map[key]["face_names"]
                            + result.get("face_names", [])
                        )
                    )
                if result.get("speaker_name"):
                    score_map[key]["speaker_name"] = result["speaker_name"]

        # Sort by fused score
        fused = list(score_map.values())
        fused.sort(key=lambda x: x["fused_score"], reverse=True)

        return fused[:limit]
