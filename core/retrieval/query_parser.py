from __future__ import annotations

from pydantic import BaseModel, Field

from core.llm.text_factory import TextLLMClient, get_text_client
from core.utils.logger import log


class SearchIntent(BaseModel):
    identity_names: list[str] = Field(default_factory=list)
    visual_description: str = ""
    temporal_clues: str = ""
    has_dialogue: bool = False


QUERY_PARSE_PROMPT = """You are a search intent parser for a video search engine.
Extract structured information from the user's search query.

Rules:
- identity_names: Extract exact names of people mentioned (e.g., "Prakash", "John")
- visual_description: Extract visual elements (actions, objects, clothing, colors)
- temporal_clues: Extract time indicators ("slow motion", "at the end", "beginning")
- has_dialogue: True if query asks about speech/dialogue content

User Query: {query}

Return ONLY valid JSON matching this schema:
{{"identity_names": [], "visual_description": "", "temporal_clues": "", "has_dialogue": false}}"""


class QueryParser:
    def __init__(self, client: TextLLMClient | None = None):
        self._client = client

    @property
    def client(self) -> TextLLMClient:
        if self._client is None:
            self._client = get_text_client()
        return self._client

    def parse(self, query: str) -> SearchIntent:
        if not query.strip():
            return SearchIntent()

        prompt = QUERY_PARSE_PROMPT.format(query=query)
        result = self.client.generate_json(prompt, SearchIntent)

        if result is None:
            log(f"Query parse fallback for: {query}")
            return self._fallback_parse(query)

        return result

    def _fallback_parse(self, query: str) -> SearchIntent:
        words = query.lower().split()
        names = []
        visual_parts = []
        temporal = ""
        has_dialogue = False

        temporal_keywords = {"beginning", "start", "end", "middle", "slow", "motion", "fast"}
        dialogue_keywords = {"said", "says", "saying", "spoke", "speaks", "talking", "dialogue", "speech"}

        for word in words:
            if word[0].isupper() if word else False:
                names.append(word.capitalize())
            elif word in temporal_keywords:
                temporal += f" {word}"
            elif word in dialogue_keywords:
                has_dialogue = True
            else:
                visual_parts.append(word)

        words_orig = query.split()
        for w in words_orig:
            if w[0].isupper() and w.lower() not in {"the", "a", "an", "in", "on", "at", "to", "for", "with", "is", "are"}:
                if w not in names:
                    names.append(w)

        return SearchIntent(
            identity_names=names[:3],
            visual_description=" ".join(visual_parts),
            temporal_clues=temporal.strip(),
            has_dialogue=has_dialogue,
        )
