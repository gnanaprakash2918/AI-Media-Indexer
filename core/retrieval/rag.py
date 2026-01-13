"""VideoRAG Orchestrator - Cognitive Search for Video.

Implements the Retrieve-Process-Answer loop for intelligent video search:
1. Query Decomposition: Parse natural language into StructuredQuery
2. Multi-Modal Search: Search visual, audio, and identity indexes
3. External Enrichment: Fetch external knowledge when needed
4. Answer Generation: Generate text answers with citations (for questions)

This is the "High IQ" search layer that makes queries like
"Why did he cry?" or "Find Prakash at the bowling alley" work correctly.
"""

from __future__ import annotations

import asyncio
import time
from typing import Any

from qdrant_client.http import models

from core.processing.enrichment import enricher
from core.retrieval.schemas import (
    QueryModality,
    SearchResultItem,
    StructuredQuery,
    VideoRAGResponse,
)
from core.storage.db import VectorDB
from core.utils.logger import log
from llm.factory import LLMFactory
from llm.interface import LLMInterface

# Query Decomposition Prompt
QUERY_DECOMPOSITION_PROMPT = """You are a VideoRAG query analyzer. Given a user query about video content,
decompose it into structured components for multi-modal search.

Analyze the query and extract:
1. VISUAL CUES: Things you can see (objects, actions, colors, clothing, locations)
2. AUDIO CUES: Things you can hear (dialogue, sounds, music)
3. TEXT CUES: On-screen text, names, titles
4. IDENTITIES: Names of specific people
5. TEMPORAL CUES: Time references (before, after, slowly, quickly)
6. IS QUESTION: Is this a question requiring an answer, or a search for clips?
7. NEEDS EXTERNAL: Does this need web search for context? (unknown people, locations, topics)

Return a JSON object with these fields:
{
  "visual_cues": ["list of visual descriptions"],
  "audio_cues": ["list of audio/dialogue cues"],
  "text_cues": ["list of text to match"],
  "identities": ["list of person names"],
  "temporal_cues": ["list of time constraints"],
  "is_question": true/false,
  "requires_external_knowledge": true/false,
  "scene_description": "dense combined description for semantic search"
}

USER QUERY: {query}

Return ONLY valid JSON, no explanation."""


# Answer Generation Prompt
ANSWER_GENERATION_PROMPT = """You are a video analysis assistant. Based on the retrieved video clips,
answer the user's question with specific citations.

QUESTION: {question}

RETRIEVED CLIPS:
{context}

Instructions:
1. Answer the question based ONLY on the provided clips
2. Cite specific timestamps and video names
3. If you cannot answer from the clips, say so
4. Be concise but complete

ANSWER:"""


class QueryDecoupler:
    """Decomposes complex queries into structured multi-modal components.

    Uses LLM to parse natural language queries like:
    "Prakash bowling at night wearing blue"

    Into structured queries with:
    - visual_cues: ["bowling alley", "night", "blue clothing"]
    - identities: ["Prakash"]
    - modalities: [VISUAL, IDENTITY]
    """

    def __init__(self, llm: LLMInterface | None = None) -> None:
        """Initializes the QueryDecoupler.

        Args:
            llm: Optional LLM interface for query decomposition. If not provided,
                the default LLM from factory will be used.
        """
        self.llm = llm or LLMFactory.get_default_llm()

    async def decompose(self, query: str) -> StructuredQuery:
        """Decomposes a natural language query into structured multi-modal components.

        Args:
            query: The natural language search query.

        Returns:
            A StructuredQuery object containing decomposed cues and metadata.
        """
        start = time.time()

        # Try LLM decomposition
        try:
            prompt = QUERY_DECOMPOSITION_PROMPT.format(query=query)
            prompt = QUERY_DECOMPOSITION_PROMPT.format(query=query)
            response = await self.llm.generate(prompt, max_tokens=800)

            # Parse JSON response
            import json

            # Find JSON in response (handle markdown code blocks)
            text = response.strip()
            if "```json" in text:
                text = text.split("```json")[1].split("```")[0]
            elif "```" in text:
                text = text.split("```")[1].split("```")[0]

            data = json.loads(text)

            # Build StructuredQuery
            structured = StructuredQuery(
                original_query=query,
                visual_cues=data.get("visual_cues", []),
                audio_cues=data.get("audio_cues", []),
                text_cues=data.get("text_cues", []),
                identities=data.get("identities", []),
                temporal_cues=data.get("temporal_cues", []),
                is_question=data.get("is_question", False),
                requires_external_knowledge=data.get(
                    "requires_external_knowledge", False
                ),
                scene_description=data.get("scene_description", query),
                decomposition_confidence=0.9,
            )

            # Set modalities based on cues
            modalities = []
            if structured.visual_cues:
                modalities.append(QueryModality.VISUAL)
            if structured.audio_cues:
                modalities.append(QueryModality.AUDIO)
            if structured.identities:
                modalities.append(QueryModality.IDENTITY)
            if structured.text_cues:
                modalities.append(QueryModality.TEXT)

            structured.modalities = modalities or [QueryModality.VISUAL]

            log(
                f"Query decomposed in {(time.time() - start) * 1000:.0f}ms: {len(modalities)} modalities"
            )
            return structured

        except Exception as e:
            log(
                f"Query decomposition failed, using fallback: {e}",
                level="WARNING",
            )
            # Fallback: use query as-is
            return StructuredQuery(
                original_query=query,
                visual_cues=[query],
                scene_description=query,
                modalities=[QueryModality.VISUAL],
                decomposition_confidence=0.5,
            )


class VideoRAGOrchestrator:
    """Main orchestrator for cognitive video search.

    Implements the full VideoRAG pipeline:
    1. Decompose query into structured components
    2. Enrich with external knowledge if needed
    3. Search across modalities
    4. Fuse and rerank results
    5. Generate answer if question
    """

    def __init__(
        self,
        db: VectorDB | None = None,
        llm: LLMInterface | None = None,
    ) -> None:
        """Initializes the VideoRAG orchestrator.

        Args:
            db: Optional vector database interface.
            llm: Optional LLM interface for enrichment and answer generation.
        """
        self.db = db or VectorDB()
        self.llm = llm or LLMFactory.get_default_llm()
        self.decoupler = QueryDecoupler(self.llm)

    async def search(
        self,
        query: str,
        limit: int = 20,
        video_path: str | None = None,
        enable_enrichment: bool = True,
        enable_answer_generation: bool = True,
    ) -> VideoRAGResponse:
        """Performs a comprehensive VideoRAG search.

        Orchestrates query decomposition, multi-modal search, external knowledge
        enrichment, and optional natural language answer generation.

        Args:
            query: The user's natural language query.
            limit: Maximum number of search results to return.
            video_path: Optional filter for a specific video.
            enable_enrichment: Whether to fetch external context for unknown entities.
            enable_answer_generation: Whether to generate a summary answer for questions.

        Returns:
            A VideoRAGResponse containing structured results and an optional answer.
        """
        total_start = time.time()

        # Step 1: Decompose query
        decomp_start = time.time()
        structured = await self.decoupler.decompose(query)
        decomp_time = (time.time() - decomp_start) * 1000

        # Step 2: External enrichment (if needed and enabled)
        external_context: dict[str, Any] = {}
        if (
            enable_enrichment
            and structured.requires_external_knowledge
            and enricher.is_available
        ):
            try:
                # Enrich identities we don't know
                for identity in structured.identities:
                    # Check if identity is in our database
                    # If not, try to enrich externally
                    enrichment = await enricher.enrich_unknown_face(
                        context=query,
                        image_description=identity,
                    )
                    if enrichment.get("possible_matches"):
                        external_context[identity] = enrichment

                # Enrich topics
                if structured.visual_cues:
                    topic = " ".join(structured.visual_cues[:3])
                    topic_context = await enricher.enrich_topic(topic)
                    if topic_context.get("context"):
                        external_context["topic_context"] = topic_context

            except Exception as e:
                log(f"External enrichment failed: {e}", level="WARNING")

        # Step 3: Multi-modal search
        search_start = time.time()
        results = await self._search_multimodal(structured, limit, video_path)
        search_time = (time.time() - search_start) * 1000

        # Step 4: Generate answer (if question)
        answer = None
        answer_citations: list[str] = []
        answer_confidence = 0.0

        if enable_answer_generation and structured.is_question and results:
            try:
                (
                    answer,
                    answer_citations,
                    answer_confidence,
                ) = await self._generate_answer(query, results[:5])
            except Exception as e:
                log(f"Answer generation failed: {e}", level="WARNING")

        total_time = (time.time() - total_start) * 1000

        return VideoRAGResponse(
            query=structured,
            results=results,
            total_results=len(results),
            answer=answer,
            answer_citations=answer_citations,
            answer_confidence=answer_confidence,
            external_context=external_context,
            decomposition_time_ms=decomp_time,
            search_time_ms=search_time,
            total_time_ms=total_time,
        )

    async def _search_multimodal(
        self,
        structured: StructuredQuery,
        limit: int,
        video_path: str | None,
    ) -> list[SearchResultItem]:
        """Executes a multi-modal search and fuses the results.

        Uses either strict intersection (for queries with clear visual and audio cues)
        or standard hybrid search (vector similarity + metadata filters).

        Args:
            structured: The decomposed structured query object.
            limit: Maximum number of results to return.
            video_path: Optional filter for a specific video.

        Returns:
            A list of fused search result items.
        """
        # Resolve identities to cluster IDs
        face_cluster_ids: list[int] = []
        for identity in structured.identities:
            cluster_id = self._resolve_identity(identity)
            if cluster_id is not None:
                face_cluster_ids.append(cluster_id)

        # STRICT INTERSECTION LOGIC (The "Rain + Smile" Test)
        # If we have distinct Visual AND Audio cues, we fetch them separately and intersect.
        has_visual = bool(structured.visual_cues)
        has_audio = bool(structured.audio_cues)

        if has_visual and has_audio:
            log(
                f"[VideoRAG] Performing Strict Intersection: Audio({structured.audio_cues}) ∩ Visual({structured.visual_cues})"
            )

            # 1. Search Visual
            visual_query = " ".join(structured.visual_cues)
            visual_results = await asyncio.to_thread(
                self.db.search_frames_hybrid,
                query=visual_query,
                limit=limit * 2,
                video_paths=video_path,
                face_cluster_ids=face_cluster_ids or None,
            )

            # 2. Search Audio
            audio_query = " ".join(structured.audio_cues)
            audio_results = await asyncio.to_thread(
                self.db.search_frames_hybrid,
                query=audio_query,
                limit=limit * 2,
                video_paths=video_path,
                face_cluster_ids=face_cluster_ids or None,
            )

            # 3. Intersect
            intersections = self._intersect_results(
                visual_results, audio_results
            )
            log(
                f"[VideoRAG] Intersection found {len(intersections)} overlapping segments."
            )
            return intersections[:limit]

        # FALLBACK / STANDARD HYBRID SEARCH (Single Query)
        search_query = structured.scene_description or structured.original_query

        # Call hybrid search
        raw_results = await asyncio.to_thread(
            self.db.search_frames_hybrid,
            query=search_query,
            limit=limit,
            video_paths=video_path,
            face_cluster_ids=face_cluster_ids or None,
        )

        # Convert to SearchResultItem
        return self._convert_to_items(raw_results)

    def _intersect_results(
        self,
        visual_hits: list[dict],
        audio_hits: list[dict],
        window: float = 5.0,
    ) -> list[SearchResultItem]:
        """Finds overlapping timestamps between visual and audio search results.

        This ensures that multi-modal queries (e.g., "someone smiling while talking")
        return segments where both events occur within a narrow time window.

        Args:
            visual_hits: Raw hits from the visual index.
            audio_hits: Raw hits from the audio/dialogue index.
            window: Time window in seconds for a valid intersection.

        Returns:
            A list of merged search result items.
        """
        matches = []

        # Group by video for efficiency
        visual_by_vid = {}
        for h in visual_hits:
            vid = h.get("video_path", "")
            if vid not in visual_by_vid:
                visual_by_vid[vid] = []
            visual_by_vid[vid].append(h)

        for a_hit in audio_hits:
            vid = a_hit.get("video_path", "")
            if vid not in visual_by_vid:
                continue

            a_time = a_hit.get("timestamp", 0.0)

            # Check for overlap with any visual hit in same video
            for v_hit in visual_by_vid[vid]:
                v_time = v_hit.get("timestamp", 0.0)

                if abs(a_time - v_time) <= window:
                    # Found intersection!
                    # Merge info
                    combined_score = (
                        a_hit.get("score", 0) + v_hit.get("score", 0)
                    ) / 2

                    # Create synthetic intersection item
                    item = SearchResultItem(
                        id=f"{a_hit['id']}_{v_hit['id']}",
                        video_path=vid,
                        timestamp=(a_time + v_time) / 2,  # Midpoint
                        score=combined_score * 1.2,  # Boost intersection
                        match_reasons=["strict_intersection"],
                        matched_entities=list(
                            set(
                                a_hit.get("entities", [])
                                + v_hit.get("entities", [])
                            )
                        ),
                        action=v_hit.get("action"),
                        dialogue=a_hit.get("dialogue") or v_hit.get("dialogue"),
                        face_names=list(
                            set(
                                a_hit.get("face_names", [])
                                + v_hit.get("face_names", [])
                            )
                        ),
                        visual_score=v_hit.get("score"),
                    )
                    matches.append(item)

        # Deduplicate matches (by approximate timestamp)
        # Sort by score desc
        matches.sort(key=lambda x: x.score, reverse=True)
        unique = []
        seen_keys = set()

        for m in matches:
            # key = video + rounded time (bucket 2s)
            key = (m.video_path, int(m.timestamp / 2.0))
            if key not in seen_keys:
                unique.append(m)
                seen_keys.add(key)

        return unique

    def _convert_to_items(
        self, raw_results: list[dict]
    ) -> list[SearchResultItem]:
        """Converts raw database results into structured SearchResultItem objects.

        Args:
            raw_results: A list of result dictionaries from the vector database.

        Returns:
            A list of initialized SearchResultItem objects.
        """
        items = []
        for r in raw_results:
            item = SearchResultItem(
                id=str(r.get("id", "")),
                video_path=r.get("video_path", ""),
                timestamp=r.get("timestamp", 0.0),
                score=r.get("score", 0.0),
                match_reasons=r.get("match_reasons", []),
                matched_entities=r.get("entities", []),
                action=r.get("action"),
                dialogue=r.get("dialogue"),
                entities=r.get("entities", []),
                face_names=r.get("face_names", []),
                visual_score=r.get("vector_score"),
                rrf_score=r.get("rrf_score"),
            )
            items.append(item)
        return items

    def _resolve_identity(self, name: str) -> int | None:
        """Resolves a person's name to a face cluster ID.

        Args:
            name: The name of the person to look up.

        Returns:
            The cluster ID if found, otherwise None.
        """
        try:
            # Search for named faces
            results = self.db.client.scroll(
                collection_name=self.db.FACES_COLLECTION,
                scroll_filter=models.Filter(
                    must=[
                        models.FieldCondition(
                            key="name", match=models.MatchValue(value=name)
                        )
                    ]
                ),
                limit=1,
                with_payload=["cluster_id"],
            )

            if results and results[0]:
                payload = results[0][0].payload or {}
                return payload.get("cluster_id")
            return None

        except Exception as e:
            log(f"Identity resolution failed: {e}", level="DEBUG")

        return None

    async def _generate_answer(
        self,
        question: str,
        results: list[SearchResultItem],
    ) -> tuple[str, list[str], float]:
        """Generates a natural language answer from top search results.

        Args:
            question: The user's original question.
            results: The top retrieved search results to use as context.

        Returns:
            A tuple containing (answer_text, citations, confidence_score).
        """
        # Build context from results
        context_parts = []
        citations = []

        for i, r in enumerate(results):
            video_name = r.video_path.split("/")[-1].split("\\")[-1]
            time_str = f"{int(r.timestamp // 60)}:{int(r.timestamp % 60):02d}"
            citation = f"[{video_name} @ {time_str}]"
            citations.append(citation)

            clip_info = f"Clip {i + 1} {citation}:"
            if r.action:
                clip_info += f" Action: {r.action}"
            if r.dialogue:
                clip_info += f' Dialogue: "{r.dialogue}"'
            if r.entities:
                clip_info += f" Entities: {', '.join(r.entities[:5])}"

            context_parts.append(clip_info)

        context = "\n".join(context_parts)

        # Generate answer
        prompt = ANSWER_GENERATION_PROMPT.format(
            question=question,
            context=context,
        )

        answer = await self.llm.generate(prompt, max_tokens=500)

        # Simple confidence based on result relevance
        avg_score = (
            sum(r.score for r in results) / len(results) if results else 0
        )
        confidence = min(0.9, avg_score + 0.2)

        return answer.strip(), citations, confidence


# Global instance (lazy initialization)
_orchestrator: VideoRAGOrchestrator | None = None


def get_orchestrator() -> VideoRAGOrchestrator:
    """Retrieves or creates the global VideoRAG orchestrator instance.

    Uses lazy initialization to ensure the orchestrator is only created when needed.

    Returns:
        The global VideoRAGOrchestrator instance.
    """
    global _orchestrator
    if _orchestrator is None:
        _orchestrator = VideoRAGOrchestrator()
    return _orchestrator
