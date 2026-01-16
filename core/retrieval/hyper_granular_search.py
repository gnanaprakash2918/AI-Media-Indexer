from __future__ import annotations

from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, Field

from core.utils.logger import get_logger

log = get_logger(__name__)

PROMPTS_DIR = Path(__file__).parent.parent.parent / "prompts"


class IdentityConstraint(BaseModel):
    name: str | None = Field(default=None)
    face_id: int | None = Field(default=None)
    voice_id: int | None = Field(default=None)
    confidence: float = Field(default=0.8, ge=0.0, le=1.0)


class ClothingConstraint(BaseModel):
    body_part: Literal["upper", "lower", "footwear", "accessory"] = "upper"
    item: str = Field(default="")
    color: str = Field(default="")
    pattern: str = Field(default="solid")
    brand: str | None = Field(default=None)
    side: Literal["left", "right", "both", "unknown"] = "unknown"


class AudioConstraint(BaseModel):
    event_class: str = Field(default="")
    min_db: int | None = Field(default=None)
    max_db: int | None = Field(default=None)
    loudness_category: str | None = Field(default=None)


class TemporalConstraint(BaseModel):
    constraint_type: Literal["delay", "duration", "sequence", "before", "after"] = "delay"
    min_ms: int | None = Field(default=None)
    max_ms: int | None = Field(default=None)
    reference_event: str | None = Field(default=None)


class TextConstraint(BaseModel):
    text: str = Field(default="")
    is_exact: bool = Field(default=False)
    location: Literal["sign", "shirt", "screen", "any"] = "any"


class SpatialConstraint(BaseModel):
    measurement_type: Literal["distance", "size", "height", "area"] = "distance"
    value_cm: float | None = Field(default=None)
    min_cm: float | None = Field(default=None)
    max_cm: float | None = Field(default=None)
    reference: str = Field(default="")


class ActionConstraint(BaseModel):
    action: str = Field(default="")
    intensity: Literal["slow", "normal", "fast", "unknown"] = "unknown"
    result: str | None = Field(default=None)


class HyperGranularQuery(BaseModel):
    original_query: str
    identities: list[IdentityConstraint] = Field(default_factory=list)
    clothing: list[ClothingConstraint] = Field(default_factory=list)
    audio: list[AudioConstraint] = Field(default_factory=list)
    temporal: list[TemporalConstraint] = Field(default_factory=list)
    text: list[TextConstraint] = Field(default_factory=list)
    spatial: list[SpatialConstraint] = Field(default_factory=list)
    actions: list[ActionConstraint] = Field(default_factory=list)
    scene_description: str = Field(default="")
    reasoning: list[str] = Field(default_factory=list)

    def constraint_count(self) -> int:
        return (len(self.identities) + len(self.clothing) + len(self.audio) +
                len(self.temporal) + len(self.text) + len(self.spatial) + len(self.actions))


def _load_prompt(name: str) -> str:
    from core.utils.prompt_loader import load_prompt
    return load_prompt(name)


class HyperGranularSearcher:
    def __init__(self, db: Any = None):
        self.db = db
        self._llm = None
        
        self._custom_prompt = _load_prompt("hyper_granular_decomposition")
        if not self._custom_prompt:
            raise RuntimeError(
                "CRITICAL: Prompt file 'hyper_granular_decomposition.txt' not found in prompts/ directory. "
                "Cannot initialize HyperGranularSearcher."
            )
        
        self._clothing_detector = None
        self._speed_estimator = None
        self._depth_estimator = None
        self._clock_reader = None
        self._active_speaker = None

    def _get_clothing_detector(self):
        if self._clothing_detector is None:
            from core.processing.clothing_attributes import ClothingAttributeDetector
            self._clothing_detector = ClothingAttributeDetector()
        return self._clothing_detector

    def _get_speed_estimator(self):
        if self._speed_estimator is None:
            from core.processing.speed_estimation import SpeedEstimator
            self._speed_estimator = SpeedEstimator()
        return self._speed_estimator

    def _get_depth_estimator(self):
        if self._depth_estimator is None:
            from core.processing.depth_estimation import DepthEstimator
            self._depth_estimator = DepthEstimator()
        return self._depth_estimator

    def _get_clock_reader(self):
        if self._clock_reader is None:
            from core.processing.clock_reader import ClockReader
            self._clock_reader = ClockReader()
        return self._clock_reader

    def _get_active_speaker(self):
        if self._active_speaker is None:
            from core.processing.active_speaker import ActiveSpeakerDetector
            self._active_speaker = ActiveSpeakerDetector()
        return self._active_speaker

    async def _ensure_llm(self) -> bool:
        if self._llm is not None:
            return True
        try:
            from llm.factory import LLMFactory
            self._llm = LLMFactory.create_llm(provider="ollama")
            log.info("[HyperGranular] LLM loaded")
            return True
        except Exception as e:
            log.warning(f"[HyperGranular] LLM load failed: {e}")
            return False

    async def decompose_query(self, query: str) -> HyperGranularQuery:
        result = HyperGranularQuery(original_query=query)
        result.reasoning.append(f"[1] Input: {len(query.split())} words")

        if not await self._ensure_llm():
            result.reasoning.append("[2] LLM unavailable, using fallback")
            return self._fallback_decompose(query, result)

        try:
            import json
            prompt = self._custom_prompt.format(query=query)
            response = await self._llm.generate(prompt)
            response_clean = response.replace("```json", "").replace("```", "").strip()
            data = json.loads(response_clean)

            for item in data.get("identities", []):
                result.identities.append(IdentityConstraint(**item))
            for item in data.get("clothing", []):
                result.clothing.append(ClothingConstraint(**item))
            for item in data.get("audio", []):
                result.audio.append(AudioConstraint(**item))
            for item in data.get("temporal", []):
                result.temporal.append(TemporalConstraint(**item))
            for item in data.get("text", []):
                result.text.append(TextConstraint(**item))
            for item in data.get("spatial", []):
                result.spatial.append(SpatialConstraint(**item))
            for item in data.get("actions", []):
                result.actions.append(ActionConstraint(**item))

            result.scene_description = data.get("scene_description", "")
            result.reasoning.append(f"[2] LLM extracted {result.constraint_count()} constraints")
            log.info(f"[HyperGranular] Decomposed: {result.constraint_count()} constraints")

        except Exception as e:
            log.warning(f"[HyperGranular] LLM parse failed: {e}")
            result.reasoning.append(f"[2] LLM parse failed: {e}")
            return self._fallback_decompose(query, result)

        return result

    def _fallback_decompose(self, query: str, result: HyperGranularQuery) -> HyperGranularQuery:
        words = query.lower().split()
        for i, word in enumerate(words):
            if word[0].isupper() if word else False:
                result.identities.append(IdentityConstraint(name=word))
            if word in ("blue", "red", "green", "white", "black", "yellow", "orange"):
                next_word = words[i + 1] if i + 1 < len(words) else ""
                result.clothing.append(ClothingConstraint(color=word, item=next_word))
            if "db" in word:
                try:
                    db_val = int("".join(filter(str.isdigit, word)))
                    result.audio.append(AudioConstraint(min_db=db_val))
                except ValueError:
                    pass
            if "ms" in word:
                try:
                    ms_val = int("".join(filter(str.isdigit, word)))
                    result.temporal.append(TemporalConstraint(min_ms=ms_val, max_ms=ms_val))
                except ValueError:
                    pass
        result.scene_description = query
        result.reasoning.append(f"[3] Fallback: {result.constraint_count()} constraints")
        return result

    async def search(
        self,
        query: str,
        limit: int = 20,
        video_path: str | None = None,
    ) -> list[dict]:
        decomposed = await self.decompose_query(query)
        log.info(f"[HyperGranular] Search: {decomposed.constraint_count()} constraints")

        results: list[dict] = []
        if not self.db:
            return [{"error": "No database configured", "decomposed": decomposed.model_dump()}]

        face_ids = self._resolve_identities(decomposed.identities)
        audio_events = self._get_audio_constraints(decomposed.audio)
        text_terms = [t.text for t in decomposed.text if t.text]

        try:
            search_text = decomposed.scene_description or query
            results = self.db.search_frames_hybrid(
                query=search_text,
                limit=limit * 3,
                video_paths=video_path,
                face_cluster_ids=face_ids if face_ids else None,
            )
        except Exception as e:
            log.warning(f"[HyperGranular] Search failed: {e}")
            try:
                results = self.db.search_frames(query=query, limit=limit)
            except Exception:
                pass

        results = self._filter_by_constraints(results, decomposed)
        results = self._score_results(results, decomposed)
        results.sort(key=lambda x: x.get("hyper_score", 0), reverse=True)

        for r in results[:limit]:
            r["decomposed_query"] = decomposed.model_dump()
            r["reasoning_trace"] = decomposed.reasoning

        return results[:limit]

    def _resolve_identities(self, identities: list[IdentityConstraint]) -> list[int]:
        if not self.db:
            return []
        ids = []
        for identity in identities:
            if identity.name:
                cid = self.db.fuzzy_get_cluster_id_by_name(identity.name)
                if cid:
                    ids.append(cid)
                    log.info(f"[HyperGranular] Resolved '{identity.name}' -> cluster {cid}")
            if identity.face_id:
                ids.append(identity.face_id)
        return ids

    def _get_audio_constraints(self, audio: list[AudioConstraint]) -> list[dict]:
        return [{"event": a.event_class, "min_db": a.min_db, "max_db": a.max_db} for a in audio]

    def _filter_by_constraints(
        self, results: list[dict], decomposed: HyperGranularQuery
    ) -> list[dict]:
        filtered = results
        for text_constraint in decomposed.text:
            if text_constraint.text:
                term = text_constraint.text.lower()
                filtered = [r for r in filtered if term in (r.get("ocr_text", "") or "").lower()
                           or term in (r.get("action", "") or "").lower()]
        return filtered

    def _score_results(
        self, results: list[dict], decomposed: HyperGranularQuery
    ) -> list[dict]:
        for result in results:
            base_score = result.get("score", 0.5)
            constraint_matches = 0
            total_constraints = decomposed.constraint_count()

            desc = (result.get("action", "") + " " + result.get("description", "")).lower()
            ocr = (result.get("ocr_text", "") or "").lower()
            faces = result.get("face_names", [])

            for identity in decomposed.identities:
                if identity.name and identity.name.lower() in str(faces).lower():
                    constraint_matches += 1

            for clothing in decomposed.clothing:
                if clothing.color and clothing.color.lower() in desc:
                    constraint_matches += 0.5
                if clothing.item and clothing.item.lower() in desc:
                    constraint_matches += 0.5

            for text in decomposed.text:
                if text.text and text.text.lower() in ocr:
                    constraint_matches += 1

            match_ratio = constraint_matches / total_constraints if total_constraints > 0 else 0
            result["hyper_score"] = base_score * 0.4 + match_ratio * 0.6
            result["constraint_matches"] = constraint_matches
            result["total_constraints"] = total_constraints

        return results
