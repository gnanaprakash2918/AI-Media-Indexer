"""Scene Aggregator with Global Context Vector.

Aggregates frame-level analysis into scene-level data AND generates
video-level global context for long-term understanding (18-hour problem).
"""

from __future__ import annotations

from collections import defaultdict
from typing import TYPE_CHECKING

from core.utils.logger import log

if TYPE_CHECKING:
    pass


class GlobalContextManager:
    """Manages global video-level context for long-term understanding."""

    def __init__(self):
        self.scene_summaries: list[str] = []
        self.key_entities: dict[str, int] = defaultdict(int)
        self.key_people: dict[str, int] = defaultdict(int)
        self.key_locations: dict[str, int] = defaultdict(int)
        self.total_duration: float = 0.0

    def add_scene(self, scene: dict) -> None:
        summary = scene.get("visual_summary", "")
        if summary:
            self.scene_summaries.append(summary)

        for name in scene.get("person_names", []):
            self.key_people[name] += 1

        location = scene.get("location", "")
        if location:
            self.key_locations[location] += 1

        for entity in scene.get("entities", []):
            name = entity.get("name", "") if isinstance(entity, dict) else str(entity)
            if name:
                self.key_entities[name] += 1

        self.total_duration = max(self.total_duration, scene.get("end_time", 0))

    def generate_global_summary(self, llm=None) -> str:
        """Generate video-level summary using LLM or rule-based fallback."""
        if llm and len(self.scene_summaries) > 3:
            return self._llm_summarize(llm)
        return self._rule_based_summary()

    def _rule_based_summary(self) -> str:
        parts = []

        top_people = sorted(self.key_people.items(), key=lambda x: -x[1])[:5]
        if top_people:
            names = ", ".join(n for n, _ in top_people)
            parts.append(f"Featuring: {names}")

        top_locations = sorted(self.key_locations.items(), key=lambda x: -x[1])[:3]
        if top_locations:
            locs = ", ".join(l for l, _ in top_locations)
            parts.append(f"Locations: {locs}")

        top_entities = sorted(self.key_entities.items(), key=lambda x: -x[1])[:5]
        if top_entities:
            ents = ", ".join(e for e, _ in top_entities)
            parts.append(f"Key objects: {ents}")

        if self.scene_summaries:
            parts.append(f"Total scenes: {len(self.scene_summaries)}")

        return ". ".join(parts) if parts else "Video content summary"

    def _llm_summarize(self, llm) -> str:
        combined = " | ".join(self.scene_summaries[:20])
        prompt = f"Summarize this video in 2-3 sentences:\n{combined}"
        try:
            return llm.generate_sync(prompt, max_tokens=150)
        except Exception as e:
            log(f"LLM summarization failed: {e}")
            return self._rule_based_summary()

    def to_payload(self) -> dict:
        return {
            "global_summary": self._rule_based_summary(),
            "scene_count": len(self.scene_summaries),
            "duration_seconds": self.total_duration,
            "top_people": list(self.key_people.keys())[:10],
            "top_locations": list(self.key_locations.keys())[:5],
            "top_entities": list(self.key_entities.keys())[:10],
        }


class SceneAggregator:
    """Aggregate frame-level data into scene-level storage units."""

    def __init__(self, min_confidence: float = 0.5):
        self.min_confidence = min_confidence
        self.global_context = GlobalContextManager()

    def aggregate_frames(
        self,
        frames: list[dict],
        start_time: float,
        end_time: float,
        dialogue_segments: list[dict] | None = None,
    ) -> dict:
        all_actions: list[str] = []
        all_entities: dict[str, dict] = {}
        all_face_ids: set[int] = set()
        all_person_names: set[str] = set()
        all_visible_text: set[str] = set()
        all_clothing: dict[int, dict] = {}
        all_accessories: dict[int, set[str]] = defaultdict(set)
        location_votes: dict[str, int] = defaultdict(int)
        cultural_votes: dict[str, int] = defaultdict(int)

        for frame in frames:
            action = frame.get("action") or frame.get("description", "")
            if action:
                all_actions.append(action)

            entities = frame.get("entities") or []
            for entity in entities:
                if isinstance(entity, dict):
                    name = entity.get("name", "")
                    if name and name not in all_entities:
                        all_entities[name] = entity

            face_ids = frame.get("face_cluster_ids") or []
            for fid in face_ids:
                if isinstance(fid, int):
                    all_face_ids.add(fid)

            names = frame.get("face_names") or frame.get("person_names") or []
            for name in names:
                if name:
                    all_person_names.add(name)

            texts = frame.get("visible_text") or []
            for text in texts:
                if text:
                    all_visible_text.add(text)

            structured = frame.get("structured_data") or {}
            if structured:
                for entity in structured.get("entities", []):
                    if isinstance(entity, dict):
                        category = entity.get("category", "").lower()
                        if category in ("clothing", "apparel", "wear"):
                            name = entity.get("name", "")
                            details = entity.get("visual_details", "")
                            for fid in face_ids:
                                if fid not in all_clothing:
                                    all_clothing[fid] = {
                                        "type": name,
                                        "color": self._extract_color(details),
                                    }
                        elif category in ("accessory", "accessories", "eyewear"):
                            name = entity.get("name", "")
                            for fid in face_ids:
                                all_accessories[fid].add(name)

            location = frame.get("scene_location") or structured.get("scene", {}).get(
                "location", ""
            )
            if location:
                location_votes[location] += 1

            cultural = frame.get("scene_cultural") or structured.get("scene", {}).get(
                "cultural_context", ""
            )
            if cultural:
                cultural_votes[cultural] += 1

        best_location = max(location_votes, key=location_votes.get) if location_votes else ""
        best_cultural = max(cultural_votes, key=cultural_votes.get) if cultural_votes else ""

        person_attributes = []
        for face_id in all_face_ids:
            clothing = all_clothing.get(face_id, {})
            attr = {
                "face_id": str(face_id),
                "name": None,
                "clothing_color": clothing.get("color", ""),
                "clothing_type": clothing.get("type", ""),
                "accessories": list(all_accessories.get(face_id, [])),
                "position": "",
            }
            person_attributes.append(attr)

        unique_actions = self._dedupe_actions(all_actions)
        action_sequence = self._build_action_sequence(unique_actions)

        dialogue = ""
        speaker_ids: set[int] = set()
        speaker_names: set[str] = set()
        if dialogue_segments:
            for seg in dialogue_segments:
                seg_start = seg.get("start", 0)
                seg_end = seg.get("end", 0)
                if seg_end > start_time and seg_start < end_time:
                    text = seg.get("text", "")
                    if text:
                        dialogue += text + " "
                    if seg.get("speaker_cluster_id"):
                        speaker_ids.add(seg["speaker_cluster_id"])
                    if seg.get("speaker_name"):
                        speaker_names.add(seg["speaker_name"])

        visual_summary = self._build_visual_summary(
            unique_actions, all_entities, all_person_names
        )

        scene_data = {
            "start_time": start_time,
            "end_time": end_time,
            "duration": end_time - start_time,
            "visual_summary": visual_summary,
            "actions": unique_actions,
            "action_sequence": action_sequence,
            "entities": list(all_entities.values()),
            "visible_text": list(all_visible_text),
            "face_cluster_ids": list(all_face_ids),
            "person_names": list(all_person_names),
            "person_attributes": person_attributes,
            "speaker_ids": list(speaker_ids),
            "speaker_names": list(speaker_names),
            "dialogue_transcript": dialogue.strip(),
            "location": best_location,
            "cultural_context": best_cultural,
            "frame_count": len(frames),
        }

        self.global_context.add_scene(scene_data)
        return scene_data

    def get_global_context(self) -> dict:
        return self.global_context.to_payload()

    def _extract_color(self, text: str) -> str:
        colors = [
            "red",
            "blue",
            "green",
            "yellow",
            "orange",
            "purple",
            "pink",
            "black",
            "white",
            "gray",
            "grey",
            "brown",
            "navy",
            "azure",
            "maroon",
            "beige",
            "cream",
            "gold",
            "silver",
            "teal",
            "cyan",
        ]
        text_lower = text.lower()
        for color in colors:
            if color in text_lower:
                return color
        return ""

    def _dedupe_actions(self, actions: list[str]) -> list[str]:
        if not actions:
            return []
        unique = []
        seen = set()
        for action in actions:
            key = action.lower().strip()[:50]
            if key not in seen:
                seen.add(key)
                unique.append(action)
        return unique[:10]

    def _build_action_sequence(self, actions: list[str]) -> str:
        if not actions:
            return ""
        if len(actions) == 1:
            return actions[0]
        parts = []
        for i, action in enumerate(actions[:5]):
            if i == 0:
                parts.append(f"First, {action.lower()}")
            elif i == len(actions) - 1 or i == 4:
                parts.append(f"finally {action.lower()}")
            else:
                parts.append(f"then {action.lower()}")
        return ", ".join(parts)

    def _build_visual_summary(
        self,
        actions: list[str],
        entities: dict[str, dict],
        person_names: set[str],
    ) -> str:
        parts = []
        if person_names:
            parts.append(f"People: {', '.join(person_names)}")
        if actions:
            parts.append(f"Actions: {'; '.join(actions[:3])}")
        entity_names = [e.get("name", "") for e in entities.values()][:5]
        if entity_names:
            parts.append(f"Objects: {', '.join(entity_names)}")
        return ". ".join(parts) if parts else "Scene"


def aggregate_frames_to_scene(
    frames: list[dict],
    start_time: float,
    end_time: float,
    dialogue_segments: list[dict] | None = None,
) -> dict:
    aggregator = SceneAggregator()
    return aggregator.aggregate_frames(frames, start_time, end_time, dialogue_segments)
