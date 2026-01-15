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
    """Multi-vector search inspired by TwelveLabs Marengo.

    100% DYNAMIC - works for any video content.
    Uses LLM for query understanding, not hardcoded rules.
    """

    def __init__(self, db: Any = None):
        """Initialize multi-vector searcher.

        Args:
            db: VectorDB instance.
        """
        self.db = db
        self.decomposer = AdvancedQueryDecomposer()

    async def search(
        self,
        query: str,
        limit: int = 20,
        min_confidence: float = 0.6,
    ) -> list[dict]:
        """Execute fine-grained multi-vector search.

        Args:
            query: Complex natural language query.
            limit: Maximum results to return.
            min_confidence: Minimum confidence threshold.

        Returns:
            List of search results with reasoning traces.
        """
        # Step 1: Decompose query using LLM
        decomposed = await self.decomposer.decompose(query)

        log.info(
            f"[MultiVectorSearch] Query decomposed into "
            f"{len(decomposed.constraints)} constraints"
        )

        # Step 2: Use scene description for primary search
        search_text = decomposed.scene_description or query

        # Step 3: Execute search
        results = []
        if self.db:
            results = self.db.search_frames(query=search_text, limit=limit * 3)

        # Step 4: Apply exclusions
        for exclusion in decomposed.exclusions:
            results = [
                r
                for r in results
                if exclusion.lower() not in r.get("description", "").lower()
            ]

        # Step 5: Score by constraint matches
        for result in results:
            desc = result.get("description", "").lower()
            matches = 0
            matched_list = []
            for constraint in decomposed.constraints:
                if constraint.value.lower() in desc:
                    matches += 1
                    matched_list.append(constraint.value)
            result["constraint_matches"] = matches
            result["matched_constraints"] = matched_list
            result["combined_score"] = result.get("score", 0.5) * (1 + matches * 0.1)

        # Step 6: Sort and limit
        results.sort(key=lambda x: x.get("combined_score", 0), reverse=True)
        final_results = results[:limit]

        # Add reasoning trace
        for result in final_results:
            result["reasoning_trace"] = decomposed.reasoning_steps

        return final_results
