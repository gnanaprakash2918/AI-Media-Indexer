"""Query parsing, embedding caching, and identity resolution."""

from __future__ import annotations

import asyncio
import hashlib
import time
from collections import OrderedDict
from typing import TYPE_CHECKING, Any

from core.knowledge.schemas import ParsedQuery
from core.utils.logger import log
from core.utils.observe import observe
from core.utils.prompt_loader import load_prompt

from config import settings

if TYPE_CHECKING:
    from core.storage.db import VectorDB
    from llm.interface import LLMInterface

DYNAMIC_QUERY_PROMPT = load_prompt("dynamic_query")
QUERY_EXPANSION_PROMPT = DYNAMIC_QUERY_PROMPT  # Legacy alias


class QueryParserMixin:
    """Mixin providing query parsing, caching, and identity resolution.

    Expects `self.db: VectorDB` and `self.llm: LLMInterface`.
    """

    db: VectorDB
    llm: LLMInterface

    def _init_cache(self) -> None:
        self._query_cache: OrderedDict[str, tuple[list[float], float]] = OrderedDict()
        self._cache_ttl = 3600
        self._cache_max_size = 1000

    async def _get_cached_embedding(self, query: str) -> list[float] | None:
        cache_key = hashlib.md5(query.encode()).hexdigest()
        if cache_key in self._query_cache:
            vector, timestamp = self._query_cache[cache_key]
            if time.time() - timestamp < self._cache_ttl:
                self._query_cache.move_to_end(cache_key)
                return vector
            else:
                del self._query_cache[cache_key]
        return None

    def _cache_embedding(self, query: str, vector: list[float]) -> None:
        cache_key = hashlib.md5(query.encode()).hexdigest()
        self._evict_expired_cache()
        if len(self._query_cache) >= self._cache_max_size:
            self._query_cache.popitem(last=False)
        self._query_cache[cache_key] = (vector, time.time())

    def _evict_expired_cache(self) -> None:
        now = time.time()
        expired_keys = [
            k for k, (_, ts) in self._query_cache.items()
            if now - ts >= self._cache_ttl
        ]
        for k in expired_keys:
            del self._query_cache[k]

    @observe("search_parse_query")
    async def parse_query(self, query: str) -> ParsedQuery:
        prompt = QUERY_EXPANSION_PROMPT.format(query=query)
        try:
            parsed = await asyncio.wait_for(
                self.llm.generate_structured(
                    schema=ParsedQuery,
                    prompt=prompt,
                    system_prompt="You are a search query parser. Return JSON only.",
                ),
                timeout=12.0,
            )
            log(f"[Search] Parsed query: {parsed.model_dump()}")
            return parsed
        except Exception as e:
            log(f"[Search] Query parsing failed: {e}, using raw query")
            return ParsedQuery(visual_keywords=[query])

    def _resolve_identity(self, person_name: str) -> int | None:
        try:
            cluster_id = self.db.get_cluster_id_by_name(person_name)
            if cluster_id is not None:
                return cluster_id
            cluster_id = self.db.fuzzy_get_cluster_id_by_name(person_name)
            if cluster_id:
                log(f"[Search] Fuzzy matched '{person_name}' → cluster {cluster_id}")
            return cluster_id
        except Exception:
            return None

    def _get_face_ids_for_cluster(self, cluster_id: int) -> list[str]:
        try:
            return self.db.get_face_ids_by_cluster(cluster_id)
        except Exception:
            return []

    def _get_models_for_modalities(self, modalities: list[str]) -> list[str]:
        model_map = settings.modality_model_map
        models = set()
        for modality in modalities:
            models.update(model_map.get(modality, []))
        return list(models)
