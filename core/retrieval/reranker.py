from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

from pydantic import BaseModel

from core.llm.vlm_factory import get_vlm_client, VLMClient
from core.processing.scene_detector import extract_scene_frame
from core.utils.logger import log

if TYPE_CHECKING:
    from core.retrieval.engine import SearchCandidate


class RerankScore(BaseModel):
    confidence: int = 0
    reason: str = ""


@dataclass
class RankedResult:
    candidate: "SearchCandidate"
    vlm_confidence: int
    vlm_reason: str


RERANK_PROMPT = """Look at these frames from a video segment.
User is searching for: "{query}"

Do these frames match the search query?
Consider: people shown, actions, objects, clothing, and setting.

Return ONLY JSON: {{"confidence": 0-100, "reason": "brief explanation"}}"""


class VLMReranker:
    def __init__(self, client: VLMClient | None = None):
        self._client = client
    
    @property
    def client(self) -> VLMClient:
        if self._client is None:
            self._client = get_vlm_client()
        return self._client
    
    def rerank(
        self,
        query: str,
        candidates: list["SearchCandidate"],
        max_candidates: int = 5,
    ) -> list[RankedResult]:
        results: list[RankedResult] = []
        prompt = RERANK_PROMPT.format(query=query)
        
        for candidate in candidates[:max_candidates]:
            score = self._score_candidate(candidate, prompt)
            results.append(RankedResult(
                candidate=candidate,
                vlm_confidence=score.confidence,
                vlm_reason=score.reason,
            ))
        
        results.sort(key=lambda r: r.vlm_confidence, reverse=True)
        return results
    
    def _score_candidate(self, candidate: "SearchCandidate", prompt: str) -> RerankScore:
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
                return RerankScore(confidence=30, reason="Parse failed, assuming partial match")
        except Exception as e:
            log(f"VLM rerank error: {e}")
            return RerankScore(confidence=0, reason=str(e)[:50])
    
    def _extract_keyframes(self, candidate: "SearchCandidate") -> list[bytes]:
        frames = []
        path = Path(candidate.video_path)
        if not path.exists():
            return frames
        
        duration = candidate.end_time - candidate.start_time
        timestamps = [
            candidate.start_time,
            candidate.start_time + duration / 2,
            candidate.end_time - 0.1 if duration > 0.2 else candidate.end_time,
        ]
        
        for ts in timestamps:
            frame_bytes = extract_scene_frame(path, ts)
            if frame_bytes:
                frames.append(frame_bytes)
        
        return frames
