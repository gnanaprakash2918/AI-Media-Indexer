from __future__ import annotations

import hashlib
from dataclasses import dataclass, field

from core.retrieval.calibration import (
    combine_scores,
    generate_thumbnail_url,
    normalize_vector_score,
)
from core.retrieval.query_parser import QueryParser, SearchIntent
from core.retrieval.reranker import RankedResult, VLMReranker
from core.schemas import MatchReason, SearchResultDetail
from core.storage.identity_graph import identity_graph
from core.utils.logger import log


@dataclass
class SearchCandidate:
    video_path: str
    start_time: float
    end_time: float
    score: float
    payload: dict = field(default_factory=dict)


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
    ) -> list[SearchResultDetail]:
        intent = self._parser.parse(query)
        log(f"Parsed intent: {intent}")

        identity_matched = bool(intent.identity_names)
        video_filter = self._get_identity_filter(intent.identity_names)
        candidates = await self._vector_search(intent, video_filter, limit)

        if not candidates:
            return []

        if use_rerank:
            ranked = self._reranker.rerank(query, candidates, max_candidates=5)
            return self._ranked_to_results(ranked, intent, identity_matched)

        return self._candidates_to_results(candidates, intent, identity_matched)

    def _get_identity_filter(self, names: list[str]) -> list[str] | None:
        if not names:
            return None

        video_ids = set()
        for name in names:
            identity = identity_graph.get_identity_by_name(name)
            if identity:
                tracks = identity_graph.get_face_tracks_for_identity(identity.id)
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
            # results = self.db.search_media_frames(  # Old invalid method
            results = self.db.search_frames_hybrid(
                query=search_text,
                limit=limit,
                video_paths=video_filter,
            )
        except Exception as e:
            log(f"Vector search error: {e}")
            return []

        candidates = []
        for r in results:
            if isinstance(r, dict):
                payload = r
                score = r.get("score", 0.5)
            else:
                payload = getattr(r, "payload", {})
                score = getattr(r, "score", 0.5)

            video_path = payload.get("video_path", payload.get("media_path", ""))
            timestamp = payload.get("timestamp", 0.0)

            candidates.append(
                SearchCandidate(
                    video_path=video_path,
                    start_time=max(0, timestamp - 2.0),
                    end_time=timestamp + 5.0,
                    score=float(score),
                    payload=payload,
                )
            )

        return candidates

    def _generate_video_id(self, path: str) -> str:
        return hashlib.md5(path.encode()).hexdigest()[:12]

    def _candidates_to_results(
        self,
        candidates: list[SearchCandidate],
        intent: SearchIntent,
        identity_matched: bool,
    ) -> list[SearchResultDetail]:
        results = []
        for c in candidates:
            reasons = []
            if identity_matched:
                reasons.append(MatchReason.IDENTITY_FACE)
            reasons.append(MatchReason.SEMANTIC_VISUAL)
            if intent.has_dialogue:
                reasons.append(MatchReason.SEMANTIC_AUDIO)

            normalized_score = normalize_vector_score(c.score)

            results.append(
                SearchResultDetail(
                    video_id=self._generate_video_id(c.video_path),
                    file_path=c.video_path,
                    start_time=c.start_time,
                    end_time=c.end_time,
                    score=normalized_score,
                    match_reasons=reasons,
                    thumbnail_url=generate_thumbnail_url(c.video_path, c.start_time),
                    dense_context=c.payload.get("action", ""),
                    matched_identities=intent.identity_names,
                )
            )

        return results

    def _ranked_to_results(
        self,
        ranked: list[RankedResult],
        intent: SearchIntent,
        identity_matched: bool,
    ) -> list[SearchResultDetail]:
        results = []
        for r in ranked:
            reasons = []
            if identity_matched:
                reasons.append(MatchReason.IDENTITY_FACE)
            reasons.append(MatchReason.SEMANTIC_VISUAL)
            reasons.append(MatchReason.VLM_VERIFIED)
            if intent.has_dialogue:
                reasons.append(MatchReason.SEMANTIC_AUDIO)

            combined_score = combine_scores(r.candidate.score, r.vlm_confidence)

            results.append(
                SearchResultDetail(
                    video_id=self._generate_video_id(r.candidate.video_path),
                    file_path=r.candidate.video_path,
                    start_time=r.candidate.start_time,
                    end_time=r.candidate.end_time,
                    score=combined_score,
                    match_reasons=reasons,
                    explanation=r.vlm_reason,
                    thumbnail_url=generate_thumbnail_url(
                        r.candidate.video_path, r.candidate.start_time
                    ),
                    dense_context=r.candidate.payload.get("action", ""),
                    matched_identities=intent.identity_names,
                )
            )

        return results


def get_search_engine(db=None) -> SearchEngine:
    return SearchEngine(db=db)
