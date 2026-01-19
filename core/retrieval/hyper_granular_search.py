"""Hyper-granular search implementation using hybrid retrieval and VLM verification."""
from __future__ import annotations

from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, Field

from core.utils.logger import get_logger

log = get_logger(__name__)

PROMPTS_DIR = Path(__file__).parent.parent.parent / "prompts"

# Cascade filtering: Physics modules (TimeSformer/Depth/RAFT) should only run
# on top-N candidates from vector search. This prevents OOM on large result sets.
PHYSICS_VERIFICATION_LIMIT = 20


class IdentityConstraint(BaseModel):
    """Filter for a specific person or face ID."""
    name: str | None = Field(default=None)
    face_id: int | None = Field(default=None)
    voice_id: int | None = Field(default=None)
    confidence: float = Field(default=0.8, ge=0.0, le=1.0)


class ClothingConstraint(BaseModel):
    """Filter for specific clothing attributes."""
    body_part: Literal["upper", "lower", "footwear", "accessory"] = "upper"
    item: str = Field(default="")
    color: str = Field(default="")
    pattern: str = Field(default="solid")
    brand: str | None = Field(default=None)
    side: Literal["left", "right", "both", "unknown"] = "unknown"


class AudioConstraint(BaseModel):
    """Filter for specific audio events or volume levels."""
    event_class: str = Field(default="")
    min_db: int | None = Field(default=None)
    max_db: int | None = Field(default=None)
    loudness_category: str | None = Field(default=None)


class TemporalConstraint(BaseModel):
    """Filter for temporal relationships between events."""
    constraint_type: Literal[
        "delay", "duration", "sequence", "before", "after"
    ] = "delay"
    min_ms: int | None = Field(default=None)
    max_ms: int | None = Field(default=None)
    reference_event: str | None = Field(default=None)


class TextConstraint(BaseModel):
    """Filter for OCR or ASR text presence."""
    text: str = Field(default="")
    is_exact: bool = Field(default=False)
    location: Literal["sign", "shirt", "screen", "any"] = "any"


class SpatialConstraint(BaseModel):
    """Filter for object size, position, or distance."""
    measurement_type: Literal["distance", "size", "height", "area"] = "distance"
    value_cm: float | None = Field(default=None)
    min_cm: float | None = Field(default=None)
    max_cm: float | None = Field(default=None)
    reference: str = Field(default="")


class ActionConstraint(BaseModel):
    """Filter for specific visual actions or motion speed."""
    action: str = Field(default="")
    intensity: Literal["slow", "normal", "fast", "unknown"] = "unknown"
    result: str | None = Field(default=None)


class HyperGranularQuery(BaseModel):
    """Decomposed query with structured constraints."""
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
        """Return total number of active constraints."""
        return (
            len(self.identities)
            + len(self.clothing)
            + len(self.audio)
            + len(self.temporal)
            + len(self.text)
            + len(self.spatial)
            + len(self.actions)
        )


def _load_prompt(name: str) -> str:
    from core.utils.prompt_loader import load_prompt

    return load_prompt(name)


class HyperGranularSearcher:
    """Orchestrator for multi-stage granular search."""
    def __init__(self, db: Any = None):
        """Initialize the granular searcher."""
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
        self._temporal_analyzer = None

    def _get_clothing_detector(self):
        if self._clothing_detector is None:
            from core.processing.clothing_attributes import (
                ClothingAttributeDetector,
            )

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

    def _get_temporal_analyzer(self):
        if self._temporal_analyzer is None:
            from core.processing.temporal import TemporalAnalyzer

            self._temporal_analyzer = TemporalAnalyzer()
        return self._temporal_analyzer

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
        """Use LLM to break down natural language into structured constraints.

        Args:
            query: The user's natural language search query.

        Returns:
            Decomposed structured query object.
        """
        result = HyperGranularQuery(original_query=query)
        result.reasoning.append(f"[1] Input: {len(query.split())} words")

        if not await self._ensure_llm() or self._llm is None:
            result.reasoning.append("[2] LLM unavailable, using fallback")
            return self._fallback_decompose(query, result)

        try:
            import json

            prompt = self._custom_prompt.format(query=query)
            response = await self._llm.generate(prompt)
            response_clean = (
                response.replace("```json", "").replace("```", "").strip()
            )
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
            result.reasoning.append(
                f"[2] LLM extracted {result.constraint_count()} constraints"
            )
            log.info(
                f"[HyperGranular] Decomposed: {result.constraint_count()} constraints"
            )

        except Exception as e:
            log.warning(f"[HyperGranular] LLM parse failed: {e}")
            result.reasoning.append(f"[2] LLM parse failed: {e}")
            return self._fallback_decompose(query, result)

        return result

    def _fallback_decompose(
        self, query: str, result: HyperGranularQuery
    ) -> HyperGranularQuery:
        words = query.lower().split()
        for i, word in enumerate(words):
            if word[0].isupper() if word else False:
                result.identities.append(IdentityConstraint(name=word))
            if word in (
                "blue",
                "red",
                "green",
                "white",
                "black",
                "yellow",
                "orange",
            ):
                next_word = words[i + 1] if i + 1 < len(words) else ""
                result.clothing.append(
                    ClothingConstraint(color=word, item=next_word)
                )
            if "db" in word:
                try:
                    db_val = int("".join(filter(str.isdigit, word)))
                    result.audio.append(AudioConstraint(min_db=db_val))
                except ValueError:
                    pass
            if "ms" in word:
                try:
                    ms_val = int("".join(filter(str.isdigit, word)))
                    result.temporal.append(
                        TemporalConstraint(min_ms=ms_val, max_ms=ms_val)
                    )
                except ValueError:
                    pass
        result.scene_description = query
        result.reasoning.append(
            f"[3] Fallback: {result.constraint_count()} constraints"
        )
        return result

    async def search(
        self,
        query: str,
        limit: int = 20,
        video_path: str | None = None,
    ) -> list[dict]:
        """Perform recursive structured search with verification.

        Args:
            query: The natural language search query.
            limit: Maximum results to return.
            video_path: Optional path to filter results by a specific video.

        Returns:
            List of matching frame results.
        """
        decomposed = await self.decompose_query(query)
        log.info(
            f"[HyperGranular] Search: {decomposed.constraint_count()} constraints"
        )

        results: list[dict] = []
        if not self.db:
            return [
                {
                    "error": "No database configured",
                    "decomposed": decomposed.model_dump(),
                }
            ]

        face_ids = self._resolve_identities(decomposed.identities)
        self._get_audio_constraints(decomposed.audio)
        [t.text for t in decomposed.text if t.text]

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

    def _resolve_identities(
        self, identities: list[IdentityConstraint]
    ) -> list[int]:
        if not self.db:
            return []
        ids = []
        for identity in identities:
            if identity.name:
                cid = self.db.fuzzy_get_cluster_id_by_name(identity.name)
                if cid:
                    ids.append(cid)
                    log.info(
                        f"[HyperGranular] Resolved '{identity.name}' -> cluster {cid}"
                    )
            if identity.face_id:
                ids.append(identity.face_id)
        return ids

    def _get_audio_constraints(
        self, audio: list[AudioConstraint]
    ) -> list[dict]:
        return [
            {"event": a.event_class, "min_db": a.min_db, "max_db": a.max_db}
            for a in audio
        ]

    def _filter_by_constraints(
        self, results: list[dict], decomposed: HyperGranularQuery
    ) -> list[dict]:
        filtered = results
        for text_constraint in decomposed.text:
            if text_constraint.text:
                term = text_constraint.text.lower()
                filtered = [
                    r
                    for r in filtered
                    if term in (r.get("ocr_text", "") or "").lower()
                    or term in (r.get("action", "") or "").lower()
                ]
        return filtered

    def _score_results(
        self, results: list[dict], decomposed: HyperGranularQuery
    ) -> list[dict]:
        for result in results:
            base_score = result.get("score", 0.5)
            constraint_matches = 0
            total_constraints = decomposed.constraint_count()

            desc = (
                result.get("action", "") + " " + result.get("description", "")
            ).lower()
            ocr = (result.get("ocr_text", "") or "").lower()
            faces = result.get("face_names", [])

            for identity in decomposed.identities:
                if (
                    identity.name
                    and identity.name.lower() in str(faces).lower()
                ):
                    constraint_matches += 1

            for clothing in decomposed.clothing:
                if clothing.color and clothing.color.lower() in desc:
                    constraint_matches += 0.5
                if clothing.item and clothing.item.lower() in desc:
                    constraint_matches += 0.5

            for text in decomposed.text:
                if text.text and text.text.lower() in ocr:
                    constraint_matches += 1

            for action in decomposed.actions:
                if action.action:
                    action_lower = action.action.lower()
                    if action_lower in desc:
                        constraint_matches += 1
                    elif any(kw in desc for kw in action_lower.split()):
                        constraint_matches += 0.5
                    result["matched_action"] = action.action

            match_ratio = (
                constraint_matches / total_constraints
                if total_constraints > 0
                else 0
            )
            result["hyper_score"] = base_score * 0.4 + match_ratio * 0.6
            result["constraint_matches"] = constraint_matches
            result["total_constraints"] = total_constraints

        return results

    async def verify_action_with_frames(
        self,
        frames: list,
        action_constraint: ActionConstraint,
        threshold: float = 0.3,
    ) -> dict:
        """Verify if a specific action occurs across a sequence of frames.

        Args:
            frames: List of candidate frames.
            action_constraint: The action to verify.
            threshold: Confidence threshold.

        Returns:
            Verification result with confidence and detected labels.
        """
        if not action_constraint.action or len(frames) < 4:
            return {"verified": False, "reason": "insufficient data"}

        try:
            analyzer = self._get_temporal_analyzer()
            actions = await analyzer.analyze_clip(
                frames, top_k=5, threshold=0.1
            )

            target = action_constraint.action.lower()
            target_words = set(target.split())

            for detected in actions:
                detected_label = detected["action"].lower().replace("_", " ")
                detected_words = set(detected_label.split())

                if target in detected_label or detected_label in target:
                    return {
                        "verified": True,
                        "detected_action": detected["action"],
                        "confidence": detected["confidence"],
                        "method": "exact_match",
                    }

                overlap = len(target_words & detected_words)
                if overlap > 0 and detected["confidence"] >= threshold:
                    return {
                        "verified": True,
                        "detected_action": detected["action"],
                        "confidence": detected["confidence"],
                        "word_overlap": overlap,
                        "method": "partial_match",
                    }

            return {
                "verified": False,
                "detected_actions": [a["action"] for a in actions],
                "target": action_constraint.action,
            }

        except Exception as e:
            log.warning(f"[HyperGranular] Action verification failed: {e}")
            return {"verified": False, "error": str(e)}

    async def scene_chain_search(
        self,
        scene_descriptions: list[str],
        max_gap_seconds: float = 60.0,
        video_path: str | None = None,
        limit: int = 10,
    ) -> list[dict]:
        """Search for a temporal sequence of scenes (A → B → C).

        This is a core GraphRAG capability - finding narrative patterns
        across time, like "Person A argues → storms out → slams door".

        Args:
            scene_descriptions: Ordered list of scene descriptions to find.
            max_gap_seconds: Maximum time gap between consecutive scenes.
            video_path: Optional filter to specific video.
            limit: Maximum sequence matches to return.

        Returns:
            List of matched scene chains with temporal metadata.
        """
        if not self.db or len(scene_descriptions) < 2:
            return []

        log.info(f"[SceneChain] Searching for {len(scene_descriptions)}-step sequence")

        # Search for each scene in the chain
        scene_candidates: list[list[dict]] = []
        for i, desc in enumerate(scene_descriptions):
            try:
                results = self.db.search_scenes(
                    query=desc,
                    limit=50,  # Get enough candidates
                    video_path=video_path,
                    search_mode="hybrid",
                )
                # Sort by start_time for temporal ordering
                results = sorted(results, key=lambda x: x.get("start_time", 0))
                scene_candidates.append(results)
                log.info(f"[SceneChain] Scene {i+1} '{desc[:30]}...': {len(results)} candidates")
            except Exception as e:
                log.warning(f"[SceneChain] Scene {i+1} search failed: {e}")
                scene_candidates.append([])

        # Find valid temporal sequences
        matched_chains = []

        # Start with first scene candidates
        if not scene_candidates or not scene_candidates[0]:
            return []

        for first_scene in scene_candidates[0]:
            chain = [first_scene]
            current_end_time = first_scene.get("end_time", first_scene.get("start_time", 0))
            video = first_scene.get("media_path", "")

            valid_chain = True
            for i in range(1, len(scene_candidates)):
                # Find next scene that follows current one within gap
                next_scene = None
                for candidate in scene_candidates[i]:
                    # Must be same video (for now)
                    if candidate.get("media_path", "") != video:
                        continue

                    candidate_start = candidate.get("start_time", 0)
                    gap = candidate_start - current_end_time

                    # Must come after current scene and within gap limit
                    if 0 <= gap <= max_gap_seconds:
                        next_scene = candidate
                        break

                if next_scene:
                    chain.append(next_scene)
                    current_end_time = next_scene.get("end_time", next_scene.get("start_time", 0))
                else:
                    valid_chain = False
                    break

            if valid_chain and len(chain) == len(scene_descriptions):
                # Calculate chain score (average of individual scores)
                chain_score = sum(s.get("score", 0) for s in chain) / len(chain)

                matched_chains.append({
                    "chain_id": len(matched_chains),
                    "video_path": video,
                    "chain_score": chain_score,
                    "start_time": chain[0].get("start_time", 0),
                    "end_time": chain[-1].get("end_time", 0),
                    "total_duration": chain[-1].get("end_time", 0) - chain[0].get("start_time", 0),
                    "scene_count": len(chain),
                    "scenes": [
                        {
                            "index": i,
                            "description": scene_descriptions[i],
                            "matched_description": s.get("description", ""),
                            "start_time": s.get("start_time", 0),
                            "end_time": s.get("end_time", 0),
                            "score": s.get("score", 0),
                            "scene_id": s.get("id", ""),
                        }
                        for i, s in enumerate(chain)
                    ],
                })

        # Sort by chain score and return top matches
        matched_chains.sort(key=lambda x: x["chain_score"], reverse=True)
        log.info(f"[SceneChain] Found {len(matched_chains)} valid sequences")
        return matched_chains[:limit]

    async def find_events_before_after(
        self,
        anchor_query: str,
        before_seconds: float = 30.0,
        after_seconds: float = 30.0,
        video_path: str | None = None,
    ) -> dict:
        """Find events that happen before and after a specific anchor event.

        Useful for cause-effect analysis: "What happened before the explosion?"

        Args:
            anchor_query: Description of the anchor event to find.
            before_seconds: How many seconds before to search.
            after_seconds: How many seconds after to search.
            video_path: Optional filter to specific video.

        Returns:
            Dict with anchor, before_events, and after_events.
        """
        if not self.db:
            return {"error": "No database configured"}

        # Find the anchor event
        try:
            anchor_results = self.db.search_scenes(
                query=anchor_query,
                limit=1,
                video_path=video_path,
                search_mode="hybrid",
            )

            if not anchor_results:
                return {"error": "Anchor event not found", "query": anchor_query}

            anchor = anchor_results[0]
            anchor_start = anchor.get("start_time", 0)
            anchor_end = anchor.get("end_time", anchor_start)
            anchor_video = anchor.get("media_path", "")

            # Get all scenes for this video
            all_scenes = self.db.get_scenes_for_video(
                video_path=anchor_video,
                limit=500,
            )

            # Find events before and after
            before_events = []
            after_events = []

            for scene in all_scenes:
                scene_start = scene.get("start_time", 0)
                scene_end = scene.get("end_time", scene_start)

                # Skip the anchor itself
                if scene.get("id") == anchor.get("id"):
                    continue

                # Check if before anchor
                if scene_end <= anchor_start and (anchor_start - scene_end) <= before_seconds:
                    before_events.append({
                        "scene_id": scene.get("id"),
                        "start_time": scene_start,
                        "end_time": scene_end,
                        "description": scene.get("description", ""),
                        "gap_to_anchor": anchor_start - scene_end,
                    })

                # Check if after anchor
                elif scene_start >= anchor_end and (scene_start - anchor_end) <= after_seconds:
                    after_events.append({
                        "scene_id": scene.get("id"),
                        "start_time": scene_start,
                        "end_time": scene_end,
                        "description": scene.get("description", ""),
                        "gap_from_anchor": scene_start - anchor_end,
                    })

            # Sort by proximity to anchor
            before_events.sort(key=lambda x: x["gap_to_anchor"])
            after_events.sort(key=lambda x: x["gap_from_anchor"])

            return {
                "anchor": {
                    "query": anchor_query,
                    "scene_id": anchor.get("id"),
                    "start_time": anchor_start,
                    "end_time": anchor_end,
                    "description": anchor.get("description", ""),
                    "video_path": anchor_video,
                },
                "before_events": before_events,
                "after_events": after_events,
                "before_count": len(before_events),
                "after_count": len(after_events),
            }

        except Exception as e:
            log.warning(f"[HyperGranular] Before/after search failed: {e}")
            return {"error": str(e)}

