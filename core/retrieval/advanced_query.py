"""Advanced Query Decomposer for Fine-Grained Video Search.

100% LLM-BASED - NO HARDCODED PATTERNS.
Prompts loaded from external files for easy customization.

Based on research from:
- Google Gemini: Temporal Query Networks (TQN), 6hr context windows
- TwelveLabs Marengo: Multi-vector per clip (2-10s segments)
- VideoRAG: Query decomposition + graph-based knowledge grounding
- GPT-4o: Frame sampling + 128K context window reasoning
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any

from core.utils.logger import get_logger

if TYPE_CHECKING:
    pass

log = get_logger(__name__)

# Prompt file path
PROMPTS_DIR = Path(__file__).parent.parent.parent / "prompts"


@dataclass
class QueryConstraint:
    """A single constraint extracted from a complex query."""

    constraint_type: str  # Dynamic type from LLM, not enum
    value: str
    attributes: dict[str, Any] = field(default_factory=dict)
    negated: bool = False
    confidence: float = 0.8

    def to_dict(self) -> dict:
        """Convert to dictionary."""
        return {
            "type": self.constraint_type,
            "value": self.value,
            "attributes": self.attributes,
            "negated": self.negated,
            "confidence": self.confidence,
        }


@dataclass
class DecomposedQuery:
    """A complex query decomposed into structured constraints."""

    original_query: str
    constraints: list[QueryConstraint] = field(default_factory=list)
    temporal_relations: list[dict] = field(default_factory=list)
    spatial_relations: list[dict] = field(default_factory=list)
    exclusions: list[str] = field(default_factory=list)
    scene_description: str = ""
    modalities_required: list[str] = field(default_factory=list)
    reasoning_steps: list[str] = field(default_factory=list)


def _load_prompt(name: str) -> str:
    """Load prompt from external file.

    Args:
        name: Prompt file name (without .txt extension).

    Returns:
        Prompt content as string.
    """
    prompt_file = PROMPTS_DIR / f"{name}.txt"
    if prompt_file.exists():
        return prompt_file.read_text(encoding="utf-8")
    else:
        log.warning(f"Prompt file not found: {prompt_file}")
        return ""


class AdvancedQueryDecomposer:
    """Decompose complex multi-paragraph queries into structured search.

    100% LLM-BASED - NO HARDCODED PATTERNS.
    Works for ANY video content worldwide.
    Prompts loaded from external files.
    """

    def __init__(self, llm_client: Any = None):
        """Initialize decomposer.

        Args:
            llm_client: LLM client for semantic parsing.
        """
        self.llm_client = llm_client
        self._llm = None
        self._prompt_template = _load_prompt("query_decomposition")

    async def _ensure_llm(self) -> bool:
        """Lazy load LLM for decomposition."""
        if self._llm is not None:
            return True

        if self.llm_client:
            self._llm = self.llm_client
            return True

        try:
            from llm.factory import LLMFactory

            self._llm = LLMFactory.create_llm(provider="ollama")
            log.info("[QueryDecomposer] Loaded LLM for decomposition")
            return True
        except Exception as e:
            log.warning(f"[QueryDecomposer] LLM load failed: {e}")
            return False

    async def decompose(self, query: str) -> DecomposedQuery:
        """Decompose a complex query using LLM - NO HARDCODING.

        Args:
            query: Complex natural language query (any length, any domain).

        Returns:
            DecomposedQuery with extracted constraints.
        """
        result = DecomposedQuery(original_query=query)
        word_count = len(query.split())
        result.reasoning_steps.append(f"[1] Input: {word_count} words")

        # Use LLM for decomposition
        if self._prompt_template and await self._ensure_llm():
            try:
                import json

                prompt = self._prompt_template.format(query=query)
                response = await self._llm.generate(prompt)

                # Parse JSON response
                response_clean = (
                    response.replace("```json", "").replace("```", "").strip()
                )
                data = json.loads(response_clean)

                # Extract constraints - FULLY DYNAMIC types from LLM
                for c in data.get("constraints", []):
                    result.constraints.append(
                        QueryConstraint(
                            constraint_type=c.get("type", "unknown"),
                            value=c.get("value", ""),
                            attributes=c.get("attributes", {}),
                            negated=c.get("negated", False),
                            confidence=c.get("confidence", 0.8),
                        )
                    )

                result.temporal_relations = data.get("temporal_relations", [])
                result.spatial_relations = data.get("spatial_relations", [])
                result.exclusions = data.get("exclusions", [])
                result.scene_description = data.get("scene_description", "")
                result.modalities_required = data.get("modalities_required", ["visual"])

                result.reasoning_steps.append(
                    f"[2] LLM extracted {len(result.constraints)} constraints"
                )
                if result.scene_description:
                    result.reasoning_steps.append(
                        f"[3] Scene: {result.scene_description[:80]}..."
                    )

                log.info(
                    f"[QueryDecomposer] LLM decomposed: {len(result.constraints)} constraints"
                )

            except Exception as e:
                log.warning(f"[QueryDecomposer] LLM decomposition failed: {e}")
                result.reasoning_steps.append(f"[2] LLM failed: {e}")

        # Fallback: Basic extraction without hardcoding
        if not result.constraints:
            result.reasoning_steps.append("[2] Using basic word extraction")
            words = query.split()
            for word in words:
                if len(word) > 4 and word[0].isupper():
                    result.constraints.append(
                        QueryConstraint(
                            constraint_type="entity",
                            value=word,
                            attributes={"source": "capitalized_word"},
                        )
                    )

        result.reasoning_steps.append(
            f"[Final] {len(result.constraints)} constraints extracted"
        )
        return result


class MultiVectorSearcher:
    """Multi-vector search for complex queries.

    Uses:
    1. LLM query decomposition
    2. Identity resolution (names → cluster IDs)
    3. Hybrid search (vector + keyword + RRF)
    4. LLM reranking with chain-of-thought
    5. Temporal windowing for context
    """

    def __init__(self, db: Any = None):
        """Initialize multi-vector searcher.

        Args:
            db: VectorDB instance.
        """
        self.db = db
        self.decomposer = AdvancedQueryDecomposer()
        self._llm = None
        self._rerank_prompt = _load_prompt("rerank_verification")

    async def _ensure_llm(self) -> bool:
        """Lazy load LLM for reranking."""
        if self._llm is not None:
            return True
        try:
            from llm.factory import LLMFactory

            self._llm = LLMFactory.create_llm(provider="ollama")
            return True
        except Exception as e:
            log.warning(f"[MultiVectorSearch] LLM load failed: {e}")
            return False

    def _resolve_identities(self, constraints: list[QueryConstraint]) -> list[int]:
        """Resolve identity constraints to cluster IDs.

        Args:
            constraints: List of query constraints.

        Returns:
            List of face cluster IDs.
        """
        if not self.db:
            return []

        cluster_ids = []
        for c in constraints:
            # Look for person/identity type constraints
            if c.constraint_type.lower() in ("person", "identity", "character", "face"):
                cid = self.db.get_cluster_id_by_name(c.value)
                if cid:
                    cluster_ids.append(cid)
                    log.info(f"[MultiVectorSearch] Resolved '{c.value}' → cluster {cid}")

        return cluster_ids

    async def search(
        self,
        query: str,
        limit: int = 20,
        min_confidence: float = 0.6,
        video_path: str | None = None,
        enable_rerank: bool = True,
    ) -> list[dict]:
        """Execute fine-grained multi-vector search.

        Args:
            query: Complex natural language query.
            limit: Maximum results to return.
            min_confidence: Minimum confidence threshold.
            video_path: Optional filter for specific video.
            enable_rerank: Whether to apply LLM reranking.

        Returns:
            List of search results with reasoning traces.
        """
        # Step 1: Decompose query using LLM
        decomposed = await self.decomposer.decompose(query)
        log.info(
            f"[MultiVectorSearch] Decomposed: {len(decomposed.constraints)} constraints"
        )

        # Step 2: Resolve identity names to cluster IDs
        face_cluster_ids = self._resolve_identities(decomposed.constraints)

        # Step 3: Build search text from scene description
        search_text = decomposed.scene_description or query

        # Step 4: Execute hybrid search (vector + keyword + RRF)
        results = []
        if self.db:
            try:
                results = self.db.search_frames_hybrid(
                    query=search_text,
                    limit=limit * 3,  # Get more for reranking
                    video_paths=video_path,
                    face_cluster_ids=face_cluster_ids if face_cluster_ids else None,
                )
            except Exception as e:
                log.warning(f"[MultiVectorSearch] Hybrid search failed: {e}")
                # Fallback to basic search
                results = self.db.search_frames(query=search_text, limit=limit * 2)

        # Step 5: Apply exclusions
        for exclusion in decomposed.exclusions:
            exclusion_lower = exclusion.lower()
            results = [
                r
                for r in results
                if exclusion_lower not in r.get("action", "").lower()
                and exclusion_lower not in r.get("description", "").lower()
            ]

        # Step 6: Score by constraint matches (pre-rerank scoring)
        for result in results:
            desc = (result.get("action", "") + " " + result.get("description", "")).lower()
            matches = 0
            matched_list = []
            for constraint in decomposed.constraints:
                if constraint.value.lower() in desc:
                    matches += 1
                    matched_list.append(constraint.value)
            result["constraint_matches"] = matches
            result["matched_constraints"] = matched_list
            base_score = result.get("rrf_score", result.get("score", 0.5))
            result["combined_score"] = base_score * (1 + matches * 0.15)

        # Step 7: LLM Reranking (if enabled and LLM available)
        if enable_rerank and results and await self._ensure_llm() and self._rerank_prompt:
            results = await self._llm_rerank(query, results[:limit * 2], decomposed)

        # Step 8: Sort by combined score and limit
        results.sort(key=lambda x: x.get("combined_score", 0), reverse=True)
        final_results = results[:limit]

        # Step 9: Add reasoning trace and metadata
        for result in final_results:
            result["reasoning_trace"] = decomposed.reasoning_steps
            result["decomposed_constraints"] = [c.to_dict() for c in decomposed.constraints]
            result["query_modalities"] = decomposed.modalities_required

        return final_results

    async def _llm_rerank(
        self,
        query: str,
        candidates: list[dict],
        decomposed: DecomposedQuery,
    ) -> list[dict]:
        """Rerank candidates using LLM chain-of-thought reasoning.

        Args:
            query: Original query.
            candidates: Candidate results to rerank.
            decomposed: Decomposed query for constraint checking.

        Returns:
            Reranked candidates with LLM scores.
        """
        import json

        reranked = []
        for candidate in candidates[:15]:  # Limit LLM calls
            description = (
                candidate.get("action", "")
                or candidate.get("description", "")
                or candidate.get("dense_caption", "")
            )

            prompt = self._rerank_prompt.format(
                query=query,
                description=description[:500],
                face_names=candidate.get("face_names", []),
                location=candidate.get("location", "unknown"),
                actions=candidate.get("action", ""),
                visible_text=candidate.get("visible_text", []),
            )

            try:
                response = await self._llm.generate(prompt)
                response_clean = response.replace("```json", "").replace("```", "").strip()
                data = json.loads(response_clean)

                llm_score = data.get("match_score", 0.5)
                candidate["llm_score"] = llm_score
                candidate["llm_reasoning"] = data.get("reasoning", "")
                candidate["llm_constraints"] = data.get("constraints_checked", [])
                candidate["llm_missing"] = data.get("missing", [])

                # Blend scores: 60% original, 40% LLM
                original_score = candidate.get("combined_score", 0.5)
                candidate["combined_score"] = original_score * 0.6 + llm_score * 0.4

                reranked.append(candidate)
            except Exception as e:
                log.debug(f"[MultiVectorSearch] Rerank failed for candidate: {e}")
                reranked.append(candidate)

        return reranked

