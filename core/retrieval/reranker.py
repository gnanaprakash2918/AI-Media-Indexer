"""Reranking Council for search result verification.

Prompts loaded from external files.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING

from pydantic import BaseModel

from core.llm.vlm_factory import VLMClient, get_vlm_client
from core.processing.scene_detector import extract_scene_frame
from core.utils.logger import get_logger
from core.utils.prompt_loader import load_prompt


log = get_logger(__name__)


class RerankScore(BaseModel):
    """Structured output for VLM re-ranking results."""

    confidence: int = 0
    reason: str = ""


@dataclass
class SearchCandidate:
    """Represents a potential search match before result calibration."""

    video_path: str
    start_time: float
    end_time: float
    score: float
    payload: dict = field(default_factory=dict)


@dataclass
class RankedResult:
    """A search result scored by the reranking council."""

    candidate: "SearchCandidate"
    final_score: float = 0.0
    vlm_confidence: int = 0
    vlm_reason: str = ""
    cross_encoder_score: float = 0.0
    bge_score: float = 0.0
    sources: dict = field(default_factory=dict)


# Load from external file
RERANK_PROMPT = load_prompt("rerank_frames")


class RerankingCouncil:
    """Multi-model reranking council per AGENTS.MD.

    Combines:
    - Cross-encoder (ms-marco-MiniLM) for query-document relevance
    - BGE-Reranker v2 for semantic scoring
    - VLM analysis for visual verification

    Usage:
        council = RerankingCouncil(weights=(0.35, 0.35, 0.30))
        results = await council.council_rerank(query, candidates)
    """

    def __init__(
        self,
        client: VLMClient | None = None,
        weights: tuple[float, float, float] = (0.35, 0.35, 0.30),
    ):
        """Initialize reranking council.

        Args:
            client: Optional VLM client for visual verification.
            weights: (cross_encoder, bge, vlm) weights for fusion.
        """
        self._client = client
        self.weights = weights
        self._cross_encoder = None
        self._bge_reranker = None
        self._models_loaded = False

    @property
    def client(self) -> VLMClient:
        """Lazy load the VLM client."""
        if self._client is None:
            self._client = get_vlm_client()
        return self._client

    def _lazy_load_models(self) -> None:
        """Load cross-encoder and BGE models lazily."""
        if self._models_loaded:
            return

        try:
            from sentence_transformers import CrossEncoder

            self._cross_encoder = CrossEncoder(
                "cross-encoder/ms-marco-MiniLM-L-6-v2",
                max_length=512,
            )
            log.info("[RerankCouncil] Loaded cross-encoder")
        except ImportError:
            log.warning("[RerankCouncil] sentence-transformers not available")
        except Exception as e:
            log.warning(f"[RerankCouncil] Cross-encoder load failed: {e}")

        try:
            from FlagEmbedding import FlagReranker

            self._bge_reranker = FlagReranker(
                "BAAI/bge-reranker-v2-m3",
                use_fp16=True,
            )
            log.info("[RerankCouncil] Loaded BGE-Reranker")
        except ImportError:
            log.warning("[RerankCouncil] FlagEmbedding not available")
        except Exception as e:
            log.warning(f"[RerankCouncil] BGE-Reranker load failed: {e}")

        self._models_loaded = True

    def _get_text_for_ranking(self, candidate: "SearchCandidate") -> str:
        """Extract text content from candidate for ranking models."""
        payload = candidate.payload or {}
        # Try multiple possible text fields in order of preference
        text = (
            payload.get("description", "")
            or payload.get("dense_caption", "")
            or payload.get("transcript", "")
            or payload.get("text", "")
            or payload.get("scene_description", "")
            or payload.get("raw_description", "")
            or ""
        )
        # Fallback: combine available fields if primary is empty
        if not text:
            parts = []
            for key in ["actions", "entity_names", "face_names", "visible_text"]:
                val = payload.get(key, [])
                if val:
                    if isinstance(val, list):
                        parts.append(" ".join(str(v) for v in val))
                    else:
                        parts.append(str(val))
            text = " ".join(parts)
        return str(text)[:1000]  # Limit length for models

    async def council_rerank(
        self,
        query: str,
        candidates: list["SearchCandidate"],
        max_candidates: int = 10,
        use_vlm: bool = True,
    ) -> list[RankedResult]:
        """Rerank using full council with weighted RRF fusion.

        Args:
            query: Search query.
            candidates: Candidates to rerank.
            max_candidates: Max candidates to process.
            use_vlm: Whether to use VLM visual verification.

        Returns:
            List of RankedResult sorted by final_score.
        """
        self._lazy_load_models()
        candidates = candidates[:max_candidates]
        scores: dict[int, dict] = defaultdict(
            lambda: {
                "cross": 0.0,
                "bge": 0.0,
                "vlm": 0.0,
                "candidate": None,
                "vlm_reason": "",
            }
        )

        # Cross-encoder scoring
        if self._cross_encoder:
            try:
                pairs = [
                    (query, self._get_text_for_ranking(c))
                    for c in candidates
                ]
                ce_scores = self._cross_encoder.predict(list(pairs))
                for i, score in enumerate(ce_scores):
                    scores[i]["cross"] = float(score)
                    scores[i]["candidate"] = candidates[i]
            except Exception as e:
                log.warning(f"[RerankCouncil] Cross-encoder failed: {e}")

        # BGE-Reranker scoring
        if self._bge_reranker:
            try:
                pairs = [
                    (query, self._get_text_for_ranking(c))
                    for c in candidates
                ]
                bge_scores = self._bge_reranker.compute_score(pairs)
                if bge_scores is not None:
                    if isinstance(bge_scores, (int, float)):
                        bge_scores = [bge_scores]
                    for i, score in enumerate(bge_scores):
                        scores[i]["bge"] = float(score)
                        scores[i]["candidate"] = candidates[i]
            except Exception as e:
                log.warning(f"[RerankCouncil] BGE-Reranker failed: {e}")

        # VLM visual verification (optional, expensive)
        if use_vlm:
            prompt = RERANK_PROMPT.format(query=query)
            for i, candidate in enumerate(candidates):
                vlm_score = self._score_candidate(candidate, prompt)
                scores[i]["vlm"] = vlm_score.confidence / 100.0
                scores[i]["vlm_reason"] = vlm_score.reason
                scores[i]["candidate"] = candidate

        # Weighted fusion
        w_ce, w_bge, w_vlm = self.weights
        results = []
        for _i, data in scores.items():
            if data["candidate"] is None:
                continue
            final = (
                w_ce * data["cross"] + w_bge * data["bge"] + w_vlm * data["vlm"]
            )
            results.append(
                RankedResult(
                    candidate=data["candidate"],
                    final_score=final,
                    vlm_confidence=int(data["vlm"] * 100),
                    vlm_reason=data["vlm_reason"],
                    cross_encoder_score=data["cross"],
                    bge_score=data["bge"],
                    sources={
                        "cross_encoder": data["cross"],
                        "bge_reranker": data["bge"],
                        "vlm": data["vlm"],
                    },
                )
            )

        results.sort(key=lambda r: r.final_score, reverse=True)
        log.info(
            f"[RerankCouncil] Reranked {len(results)} candidates, "
            f"weights=({w_ce:.2f}, {w_bge:.2f}, {w_vlm:.2f})"
        )
        return results

    def rerank(
        self,
        query: str,
        candidates: list[SearchCandidate],
        max_candidates: int = 5,
    ) -> list[RankedResult]:
        """Re-rank candidates based on VLM visual analysis."""
        results: list[RankedResult] = []
        prompt = RERANK_PROMPT.format(query=query)

        for candidate in candidates[:max_candidates]:
            score = self._score_candidate(candidate, prompt)
            results.append(
                RankedResult(
                    candidate=candidate,
                    vlm_confidence=score.confidence,
                    vlm_reason=score.reason,
                )
            )

        results.sort(key=lambda r: r.vlm_confidence, reverse=True)
        return results

    def _score_candidate(
        self, candidate: SearchCandidate, prompt: str
    ) -> RerankScore:
        frames = self._extract_keyframes(candidate)
        if not frames:
            return RerankScore(confidence=0, reason="No frames extracted")

        best_frame = frames[len(frames) // 2] if len(frames) > 1 else frames[0]

        try:
            raw = self.client.generate_caption_from_bytes(best_frame, prompt)
            if not raw:
                return RerankScore(confidence=0, reason="VLM returned empty")

            clean = raw.strip()
            if clean.startswith("```"):
                clean = clean.split("```")[1]
                if clean.startswith("json"):
                    clean = clean[4:]

            try:
                return RerankScore.model_validate_json(clean)
            except Exception:
                start = raw.find("{")
                end = raw.rfind("}") + 1
                if start >= 0 and end > start:
                    return RerankScore.model_validate_json(raw[start:end])
                return RerankScore(
                    confidence=30, reason="Parse failed, assuming partial match"
                )
        except Exception as e:
            log.error(f"VLM rerank error: {e}")
            return RerankScore(confidence=0, reason=str(e)[:50])

    def _extract_keyframes(self, candidate: SearchCandidate) -> list[bytes]:
        frames = []
        path = Path(candidate.video_path)
        if not path.exists():
            return frames

        duration = candidate.end_time - candidate.start_time
        
        # Adaptive sampling: More frames for longer clips to catch dynamic actions
        # Min 3 frames (start, mid, end) for short clips
        # Max 8 frames for long clips to prevent OOM/timeouts
        # Stride ~3-5 seconds ideally
        
        num_frames = 3
        if duration > 5.0:
            num_frames = min(8, int(duration / 3.0))
            num_frames = max(3, num_frames)
            
        step = duration / (num_frames - 1) if num_frames > 1 else 0
        timestamps = [candidate.start_time + i * step for i in range(num_frames)]
        
        # Ensure last frame is slightly before end to avoid seeking past EOF
        timestamps[-1] = max(candidate.start_time, candidate.end_time - 0.1)

        for ts in timestamps:
            frame_bytes = extract_scene_frame(path, ts)
            if frame_bytes:
                frames.append(frame_bytes)

        return frames


# Backwards compatibility alias
VLMReranker = RerankingCouncil
