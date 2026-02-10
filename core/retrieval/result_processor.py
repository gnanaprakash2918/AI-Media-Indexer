"""Result processing: reranking, granular scoring, RRF fusion, normalization."""

from __future__ import annotations

import math
from typing import TYPE_CHECKING, Any

from pydantic import BaseModel, Field

from core.knowledge.schemas import ParsedQuery
from core.utils.logger import log
from core.utils.observe import observe
from core.utils.prompt_loader import load_prompt

if TYPE_CHECKING:
    from llm.interface import LLMInterface


class RerankResult(BaseModel):
    match_score: float = Field(default=0.5)
    constraints_checked: list[str] = Field(default_factory=list)
    reasoning: str = Field(default="")
    missing: list[str] = Field(default_factory=list)


class ResultProcessorMixin:
    """Mixin providing reranking, granular scoring, RRF fusion.

    Expects `self.llm: LLMInterface`.
    """

    llm: LLMInterface

    @observe("search_rerank_llm")
    async def rerank_with_llm(
        self,
        query: str,
        candidates: list[dict],
        top_k: int = 10,
    ) -> list[dict]:
        if not candidates:
            return []

        rerank_prompt = load_prompt("rerank_verification")
        reranked = []

        for candidate in candidates[:top_k]:
            description = (
                candidate.get("description", "")
                or candidate.get("dense_caption", "")
                or candidate.get("raw_description", "")
            )

            prompt = rerank_prompt.format(
                query=query,
                description=description[:500],
                face_names=candidate.get("face_names", []),
                location=candidate.get("location", "unknown"),
                actions=candidate.get("actions", []),
                visible_text=candidate.get("visible_text", []),
            )

            try:
                result = None
                for attempt in range(2):
                    try:
                        result = await self.llm.generate_structured(
                            schema=RerankResult,
                            prompt=prompt,
                            system_prompt="You are a video search result verifier. Return ONLY valid JSON with match_score, reasoning, constraints_checked, and missing fields.",
                        )
                        break
                    except Exception as retry_err:
                        if attempt == 0:
                            log(f"[Rerank] Retry after parse error: {retry_err}")
                            continue
                        raise

                if result is None:
                    raise ValueError("LLM returned no result after retries")

                missing_penalty = len(result.missing) * 0.1
                adjusted_score = max(0.0, result.match_score - missing_penalty)
                enriched = {**candidate}
                enriched["llm_score"] = adjusted_score
                enriched["llm_reasoning"] = result.reasoning
                enriched["constraints_satisfied"] = result.constraints_checked
                enriched["constraints_missing"] = result.missing
                from config import settings
                enriched["combined_score"] = (
                    enriched.get("score", 0) * settings.rerank_vector_weight
                    + adjusted_score * settings.rerank_llm_weight
                )
                reranked.append(enriched)

            except Exception as e:
                log(f"[Rerank] LLM verification failed: {e}")
                candidate["combined_score"] = candidate.get("score", 0) * 0.8
                candidate["llm_reasoning"] = f"Verification failed: {str(e)[:50]}"
                reranked.append(candidate)

        reranked.sort(key=lambda x: x.get("combined_score", 0), reverse=True)
        return reranked[:top_k]

    def _apply_granular_scoring(
        self, results: list[dict], parsed: ParsedQuery
    ) -> list[dict]:
        for result in results:
            base_score = float(result.get("score", 0.5))
            payload = (
                result.get("base_payload", result)
                if isinstance(result, dict)
                else result
            )

            desc = " ".join(filter(None, [
                str(payload.get("description", "")),
                str(payload.get("visual_text", "")),
                str(payload.get("visual_summary", "")),
                str(payload.get("action", "")),
                str(payload.get("motion_text", "")),
                str(payload.get("action_summary", "")),
                str(payload.get("location", "")),
                str(payload.get("cultural_context", "")),
                " ".join(
                    f"{e.get('name', '')} {e.get('visual_details', '')}"
                    for e in (payload.get("entities") or [])
                    if isinstance(e, dict)
                ),
                " ".join(payload.get("clothing_descriptions", [])),
                " ".join(payload.get("clothing_types", [])),
                " ".join(payload.get("accessories", [])),
            ])).lower()

            ocr = " ".join(filter(None, [
                str(payload.get("ocr_text", "")),
                str(payload.get("visible_text", "")),
            ])).lower()

            dialogue = " ".join(filter(None, [
                str(payload.get("dialogue_transcript", "")),
                str(payload.get("dialogue_text", "")),
            ])).lower()

            audio = " ".join(
                str(e) for e in (payload.get("audio_events") or [])
            ).lower()

            all_text = f"{desc} {ocr} {dialogue} {audio}"

            matches = 0
            total_checks = 0

            # Text constraints
            if hasattr(parsed, "text") and parsed.text:
                for t in parsed.text:
                    total_checks += 1
                    search_text = t.get("text", "").lower()
                    if search_text in ocr:
                        matches += 1
                    elif search_text in desc:
                        matches += 0.7
                    elif search_text in dialogue:
                        matches += 0.5

            # Clothing constraints
            if hasattr(parsed, "clothing") and parsed.clothing:
                for c in parsed.clothing:
                    total_checks += 1
                    c_item = c.get("item", "").lower()
                    c_color = c.get("color", "").lower()
                    item_found = c_item and c_item in desc
                    color_found = c_color and c_color in desc
                    if item_found and color_found:
                        matches += 1
                    elif item_found or color_found:
                        matches += 0.5

            # Action constraints
            if hasattr(parsed, "actions") and parsed.actions:
                for a in parsed.actions:
                    total_checks += 1
                    action_val = a.get("action", "").lower()
                    if action_val and action_val in desc:
                        matches += 1
                    result_val = a.get("result", "").lower()
                    if result_val and result_val in desc:
                        matches += 0.5

            # Location constraints
            if hasattr(parsed, "location") and parsed.location:
                total_checks += 1
                if parsed.location.lower() in desc:
                    matches += 1
                elif parsed.location.lower() in all_text:
                    matches += 0.5

            # Identity constraints
            if hasattr(parsed, "identities") and parsed.identities:
                person_names = [
                    str(n).lower()
                    for n in (payload.get("person_names") or [])
                ]
                for identity in parsed.identities:
                    id_name = identity.get("name", "").lower()
                    if id_name:
                        total_checks += 1
                        if id_name in person_names:
                            matches += 1
                        elif id_name in desc:
                            matches += 0.5

            # Audio constraints
            if hasattr(parsed, "audio") and parsed.audio:
                for aud in parsed.audio:
                    total_checks += 1
                    aud_val = aud.get("event", "").lower()
                    if aud_val and aud_val in audio:
                        matches += 1
                    elif aud_val and aud_val in all_text:
                        matches += 0.3

            # Spatial constraints
            if hasattr(parsed, "spatial") and parsed.spatial:
                for sp in parsed.spatial:
                    total_checks += 1
                    sp_type = sp.get("measurement_type", "").lower()
                    if sp_type and sp_type in desc:
                        matches += 0.5

            boost = 0.0
            if total_checks > 0:
                coverage = matches / total_checks
                boost = coverage * 0.5

            penalty = 0.0
            if hasattr(parsed, "exclusions") and parsed.exclusions:
                for exc in parsed.exclusions:
                    exc_value = exc.get("value", "").lower()
                    if exc_value and exc_value in all_text:
                        penalty += 0.3

            if isinstance(result, dict):
                result["score"] = max(0, base_score + boost - penalty)
                result["granular_matches"] = matches
                result["granular_coverage"] = matches / total_checks if total_checks > 0 else 0
                if penalty > 0:
                    result["exclusion_penalty"] = penalty

        return results

    def _normalize_score(self, score: float) -> float:
        try:
            return 1 / (1 + math.exp(-2.0 * score))
        except OverflowError:
            return 0.0 if score < 0 else 1.0

    def _rrf_fusion_multimodal(
        self,
        modality_results: dict[str, list[dict]],
        limit: int = 50,
        k: int = 60,
    ) -> list[dict]:
        score_map: dict[tuple, dict] = {}

        for modality, results in modality_results.items():
            if not results:
                continue

            for rank, result in enumerate(results):
                vp = result.get("video_path") or result.get("media_path") or ""
                st = round(
                    float(result.get("start_time", result.get("timestamp", 0))),
                    1,
                )
                key = (vp, st)
                rrf_score = 1.0 / (k + rank + 1)

                if key not in score_map:
                    score_map[key] = {
                        "video_path": vp,
                        "start_time": st,
                        "end_time": result.get("end_time", st + 5),
                        "fused_score": 0.0,
                        "modalities": [],
                        "description": result.get("description", ""),
                        "face_names": result.get("face_names", []),
                        "speaker_name": result.get("speaker_name"),
                    }

                score_map[key]["fused_score"] += rrf_score
                score_map[key]["modalities"].append(modality)

                if result.get("description") and not score_map[key]["description"]:
                    score_map[key]["description"] = result["description"]
                if result.get("face_names"):
                    score_map[key]["face_names"] = list(
                        set(score_map[key]["face_names"] + result.get("face_names", []))
                    )
                if result.get("speaker_name"):
                    score_map[key]["speaker_name"] = result["speaker_name"]

        fused = list(score_map.values())
        fused.sort(key=lambda x: x["fused_score"], reverse=True)
        return fused[:limit]

    def _score_scene_step_match(
        self,
        scene: Any,
        step: dict[str, str],
        person_cluster_map: dict[str, list[int]],
    ) -> float:
        score = 0.0
        checks = 0

        desc = (scene.description or "").lower()
        actions = " ".join(scene.actions).lower() if scene.actions else ""
        location = (scene.location or "").lower()
        all_text = f"{desc} {actions} {location}"

        person_name = step.get("person", "").strip()
        if person_name:
            checks += 1
            target_ids = person_cluster_map.get(person_name, [])
            if target_ids and any(cid in scene.face_cluster_ids for cid in target_ids):
                score += 1.0
            elif person_name.lower() in desc:
                score += 0.5

        action = step.get("action", "").strip()
        if action:
            checks += 1
            if action.lower() in actions:
                score += 1.0
            elif action.lower() in desc:
                score += 0.7

        loc = step.get("location", "").strip()
        if loc:
            checks += 1
            if loc.lower() in location:
                score += 1.0
            elif loc.lower() in desc:
                score += 0.5

        desc_query = step.get("description", "").strip()
        if desc_query:
            checks += 1
            words = desc_query.lower().split()
            word_hits = sum(1 for w in words if w in all_text)
            score += word_hits / max(len(words), 1)

        return score / max(checks, 1)
