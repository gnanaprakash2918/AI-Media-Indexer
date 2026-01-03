from __future__ import annotations
from dataclasses import dataclass, field
from typing import Any

from pydantic import BaseModel

from config import settings
from core.retrieval.query_parser import QueryParser, SearchIntent
from core.retrieval.reranker import VLMReranker, RankedResult
from core.storage.identity_graph import identity_graph
from core.utils.logger import log


@dataclass
class SearchCandidate:
    video_path: str
    start_time: float
    end_time: float
    score: float
    payload: dict = field(default_factory=dict)


class EnhancedSearchResult(BaseModel):
    video_path: str
    start_time: float
    end_time: float
    confidence: float
    modalities: list[str] = []
    matched_identities: list[str] = []
    dense_context: str = ""
    vlm_reason: str = ""


class SearchEngine:
    def __init__(self, db=None):
        self._db = db
        self._parser = QueryParser()
        self._reranker = VLMReranker()
    
    @property
    def db(self):
        if self._db is None:
            from core.storage.db import VectorDB
            self._db = VectorDB()
        return self._db
    
    async def search(
        self,
        query: str,
        use_rerank: bool = False,
        limit: int = 20,
    ) -> list[EnhancedSearchResult]:
        intent = self._parser.parse(query)
        log(f"Parsed intent: {intent}")
        
        video_filter = self._get_identity_filter(intent.identity_names)
        candidates = await self._vector_search(intent, video_filter, limit)
        
        if not candidates:
            return []
        
        if use_rerank and candidates:
            ranked = self._reranker.rerank(query, candidates, max_candidates=5)
            return self._ranked_to_results(ranked, intent)
        
        return self._candidates_to_results(candidates, intent)
    
    def _get_identity_filter(self, names: list[str]) -> list[str] | None:
        if not names:
            return None
        
        video_ids = set()
        for name in names:
            identity = identity_graph.get_identity_by_name(name)
            if identity:
                tracks = identity_graph.get_tracks_for_identity(identity.id)
                for track in tracks:
                    video_ids.add(track.media_id)
        
        if video_ids:
            log(f"Identity filter: {len(video_ids)} videos for {names}")
            return list(video_ids)
        
        log(f"No identities found for: {names}, proceeding with global search")
        return None
    
    async def _vector_search(
        self,
        intent: SearchIntent,
        video_filter: list[str] | None,
        limit: int,
    ) -> list[SearchCandidate]:
        search_text = intent.visual_description
        if intent.temporal_clues:
            search_text += f" {intent.temporal_clues}"
        if intent.has_dialogue:
            search_text += " speaking dialogue conversation"
        
        if not search_text.strip():
            search_text = "video scene"
        
        try:
            results = self.db.search_media_frames(
                query=search_text,
                limit=limit,
                video_path_filter=video_filter,
            )
        except Exception as e:
            log(f"Vector search error: {e}")
            return []
        
        candidates = []
        for r in results:
            payload = r.payload if hasattr(r, "payload") else {}
            if isinstance(r, dict):
                payload = r
            
            video_path = payload.get("video_path", payload.get("media_path", ""))
            timestamp = payload.get("timestamp", 0.0)
            score = r.score if hasattr(r, "score") else payload.get("score", 0.5)
            
            candidates.append(SearchCandidate(
                video_path=video_path,
                start_time=max(0, timestamp - 2.0),
                end_time=timestamp + 5.0,
                score=float(score),
                payload=payload,
            ))
        
        return candidates
    
    def _candidates_to_results(
        self,
        candidates: list[SearchCandidate],
        intent: SearchIntent,
    ) -> list[EnhancedSearchResult]:
        results = []
        for c in candidates:
            modalities = ["visual"]
            if intent.has_dialogue:
                modalities.append("speech")
            
            results.append(EnhancedSearchResult(
                video_path=c.video_path,
                start_time=c.start_time,
                end_time=c.end_time,
                confidence=c.score * 100,
                modalities=modalities,
                matched_identities=intent.identity_names,
                dense_context=c.payload.get("action", ""),
            ))
        
        return results
    
    def _ranked_to_results(
        self,
        ranked: list[RankedResult],
        intent: SearchIntent,
    ) -> list[EnhancedSearchResult]:
        results = []
        for r in ranked:
            modalities = ["visual", "vlm_verified"]
            if intent.has_dialogue:
                modalities.append("speech")
            
            results.append(EnhancedSearchResult(
                video_path=r.candidate.video_path,
                start_time=r.candidate.start_time,
                end_time=r.candidate.end_time,
                confidence=float(r.vlm_confidence),
                modalities=modalities,
                matched_identities=intent.identity_names,
                dense_context=r.candidate.payload.get("action", ""),
                vlm_reason=r.vlm_reason,
            ))
        
        return results


def get_search_engine(db=None) -> SearchEngine:
    return SearchEngine(db=db)
