"""LLM-Based Named Entity Recognition for Video Content.

100% LLM-BASED - NO HARDCODED KNOWLEDGE BASES.
Prompts loaded from external files for easy customization.
Works for ANY video content worldwide.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING

from core.utils.logger import get_logger

if TYPE_CHECKING:
    pass

log = get_logger(__name__)

# Prompt file path
PROMPTS_DIR = Path(__file__).parent.parent.parent / "prompts"


def _load_prompt(name: str) -> str:
    """Load prompt from external file."""
    prompt_file = PROMPTS_DIR / f"{name}.txt"
    if prompt_file.exists():
        return prompt_file.read_text(encoding="utf-8")
    log.warning(f"Prompt file not found: {prompt_file}")
    return ""


@dataclass
class Entity:
    """Extracted named entity - fully dynamic types."""

    text: str
    entity_type: str  # Dynamic type from LLM, not enum
    start: int = 0
    end: int = 0
    confidence: float = 1.0
    context: str = ""
    metadata: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        """Convert to dictionary."""
        return {
            "text": self.text,
            "type": self.entity_type,
            "start": self.start,
            "end": self.end,
            "confidence": self.confidence,
            "context": self.context,
            "metadata": self.metadata,
        }


class VideoNERExtractor:
    """Extract named entities using LLM - works for ANY video content.

    NO HARDCODED KNOWLEDGE BASES.
    Prompts loaded from external files.
    Uses LLM to identify entities dynamically.
    """

    def __init__(self, use_llm: bool = True):
        """Initialize NER extractor.

        Args:
            use_llm: Whether to use LLM for NER.
        """
        self.use_llm = use_llm
        self._llm = None
        self._prompt_template = _load_prompt("ner_extraction")

    async def _ensure_llm(self) -> bool:
        """Lazy load LLM for NER."""
        if self._llm is not None:
            return True

        try:
            from llm.factory import LLMFactory

            self._llm = LLMFactory.create_llm(provider="ollama")
            log.info("[NER] Loaded LLM for entity extraction")
            return True
        except Exception as e:
            log.warning(f"[NER] LLM load failed: {e}")
            return False

    async def extract(self, text: str) -> list[Entity]:
        """Extract all entities from text using LLM.

        Args:
            text: Input text (query, description, transcript).

        Returns:
            List of extracted Entity objects.
        """
        entities = []

        if self.use_llm and self._prompt_template and await self._ensure_llm():
            try:
                import json

                prompt = self._prompt_template.format(text=text)
                if self._llm is None:
                    return []
                response = await self._llm.generate(prompt)

                # Parse JSON response
                response_clean = (
                    response.replace("```json", "").replace("```", "").strip()
                )
                data = json.loads(response_clean)

                for item in data:
                    entity_text = item.get("text", "")
                    start = text.find(entity_text)
                    end = start + len(entity_text) if start >= 0 else 0

                    entities.append(
                        Entity(
                            text=entity_text,
                            entity_type=item.get("type", "UNKNOWN"),
                            start=start,
                            end=end,
                            confidence=item.get("confidence", 0.8),
                            context=item.get("context", ""),
                            metadata={"source": "llm"},
                        )
                    )

                log.info(f"[NER] LLM extracted {len(entities)} entities")

            except Exception as e:
                log.warning(f"[NER] LLM extraction failed: {e}")

        # Fallback: Basic extraction without hardcoding
        if not entities:
            log.info("[NER] Using basic capitalized word extraction")
            words = text.split()
            for word in words:
                clean_word = word.strip(".,!?\"'();:")
                if len(clean_word) > 2 and clean_word[0].isupper():
                    start = text.find(word)
                    entities.append(
                        Entity(
                            text=clean_word,
                            entity_type="ENTITY",
                            start=start,
                            end=start + len(clean_word),
                            confidence=0.5,
                            metadata={"source": "fallback"},
                        )
                    )

        return entities

    async def extract_for_search(self, query: str) -> dict[str, list[str]]:
        """Extract entities grouped by type for search filtering."""
        entities = await self.extract(query)

        grouped: dict[str, list[str]] = {}
        for entity in entities:
            type_name = entity.entity_type.lower()
            if type_name not in grouped:
                grouped[type_name] = []
            if entity.text not in grouped[type_name]:
                grouped[type_name].append(entity.text)

        return grouped

    async def enrich_description(
        self,
        description: str,
        entities: list[Entity] | None = None,
    ) -> str:
        """Enrich a VLM description with structured entity tags."""
        if entities is None:
            entities = await self.extract(description)

        by_type: dict[str, list[str]] = {}
        for entity in entities:
            if entity.entity_type not in by_type:
                by_type[entity.entity_type] = []
            if entity.text not in by_type[entity.entity_type]:
                by_type[entity.entity_type].append(entity.text)

        enriched_parts = [description, "\n\n[Entities]"]
        for entity_type, texts in by_type.items():
            enriched_parts.append(f"- {entity_type}: {', '.join(texts)}")

        return "\n".join(enriched_parts)


# Singleton
_ner_extractor: VideoNERExtractor | None = None


def get_ner_extractor() -> VideoNERExtractor:
    """Get singleton NER extractor instance."""
    global _ner_extractor
    if _ner_extractor is None:
        _ner_extractor = VideoNERExtractor()
    return _ner_extractor
