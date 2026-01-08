"""Scene Aggregator for production-grade video search.

Aggregates frame-by-frame analysis into coherent scene-level data.
This is critical for enabling complex queries like:
"Prakash wearing blue shirt bowling at Brunswick hitting a strike"

The aggregator:
1. Groups frames by scene boundaries
2. Deduplicates entities and actions
3. Tracks person attributes across frames
4. Generates scene summaries
"""

from __future__ import annotations

from collections import defaultdict
from typing import TYPE_CHECKING

from core.utils.logger import log

if TYPE_CHECKING:
    from core.knowledge.schemas import (
        EntityDetail,
        FrameAnalysis,
        PersonAttribute,
        SceneData,
    )


class SceneAggregator:
    """Aggregate frame-level data into scene-level storage units."""

    def __init__(self, min_confidence: float = 0.5):
        """Initialize aggregator.

        Args:
            min_confidence: Minimum confidence threshold for including entities.
        """
        self.min_confidence = min_confidence

    def aggregate_frames(
        self,
        frames: list[dict],
        start_time: float,
        end_time: float,
        dialogue_segments: list[dict] | None = None,
    ) -> dict:
        """Aggregate a list of frames into a single scene.

        Args:
            frames: List of frame data dicts (from pipeline processing).
            start_time: Scene start timestamp.
            end_time: Scene end timestamp.
            dialogue_segments: ASR segments overlapping this scene.

        Returns:
            Dict ready for SceneData creation.
        """
        # Collect all data across frames
        all_actions: list[str] = []
        all_entities: dict[str, dict] = {}  # name -> entity_detail
        all_face_ids: set[int] = set()
        all_person_names: set[str] = set()
        all_visible_text: set[str] = set()
        all_clothing: dict[int, dict] = {}  # face_id -> clothing info
        all_accessories: dict[int, set[str]] = defaultdict(set)

        # Location voting (most common wins)
        location_votes: dict[str, int] = defaultdict(int)
        cultural_votes: dict[str, int] = defaultdict(int)

        for frame in frames:
            # Extract actions
            action = frame.get("action") or frame.get("description", "")
            if action:
                all_actions.append(action)

            # Extract entities
            entities = frame.get("entities") or []
            for entity in entities:
                if isinstance(entity, dict):
                    name = entity.get("name", "")
                    if name and name not in all_entities:
                        all_entities[name] = entity

            # Extract face clusters
            face_ids = frame.get("face_cluster_ids") or []
            for fid in face_ids:
                if isinstance(fid, int):
                    all_face_ids.add(fid)

            # Extract person names
            names = frame.get("face_names") or frame.get("person_names") or []
            for name in names:
                if name:
                    all_person_names.add(name)

            # Extract visible text
            texts = frame.get("visible_text") or []
            for text in texts:
                if text:
                    all_visible_text.add(text)

            # Extract structured data for clothing
            structured = frame.get("structured_data") or {}
            if structured:
                # Parse clothing from entities
                for entity in structured.get("entities", []):
                    if isinstance(entity, dict):
                        category = entity.get("category", "").lower()
                        if category in ("clothing", "apparel", "wear"):
                            name = entity.get("name", "")
                            details = entity.get("visual_details", "")
                            # Try to associate with face
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

            # Vote for location
            location = frame.get("scene_location") or structured.get("scene", {}).get("location", "")
            if location:
                location_votes[location] += 1

            cultural = frame.get("scene_cultural") or structured.get("scene", {}).get("cultural_context", "")
            if cultural:
                cultural_votes[cultural] += 1

        # Determine winning location/cultural context
        best_location = max(location_votes, key=location_votes.get, default="")
        best_cultural = max(cultural_votes, key=cultural_votes.get, default="")

        # Build person attributes
        person_attributes = []
        for face_id in all_face_ids:
            clothing = all_clothing.get(face_id, {})
            attr = {
                "face_id": str(face_id),
                "name": None,  # Will be filled from HITL
                "clothing_color": clothing.get("color", ""),
                "clothing_type": clothing.get("type", ""),
                "accessories": list(all_accessories.get(face_id, [])),
                "position": "",
            }
            person_attributes.append(attr)

        # Deduplicate actions (keep unique, preserve order)
        unique_actions = self._dedupe_actions(all_actions)

        # Build action sequence narrative
        action_sequence = self._build_action_sequence(unique_actions)

        # Get dialogue transcript for this scene
        dialogue = ""
        speaker_ids: set[int] = set()
        speaker_names: set[str] = set()
        if dialogue_segments:
            for seg in dialogue_segments:
                seg_start = seg.get("start", 0)
                seg_end = seg.get("end", 0)
                # Check overlap with scene
                if seg_end > start_time and seg_start < end_time:
                    text = seg.get("text", "")
                    if text:
                        dialogue += text + " "
                    # Collect speaker info
                    if seg.get("speaker_cluster_id"):
                        speaker_ids.add(seg["speaker_cluster_id"])
                    if seg.get("speaker_name"):
                        speaker_names.add(seg["speaker_name"])

        # Build visual summary
        visual_summary = self._build_visual_summary(
            unique_actions, all_entities, all_person_names
        )

        return {
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

    def _extract_color(self, text: str) -> str:
        """Extract color from visual details text."""
        colors = [
            "red", "blue", "green", "yellow", "orange", "purple", "pink",
            "black", "white", "gray", "grey", "brown", "navy", "azure",
            "maroon", "beige", "cream", "gold", "silver", "teal", "cyan",
        ]
        text_lower = text.lower()
        for color in colors:
            if color in text_lower:
                return color
        return ""

    def _dedupe_actions(self, actions: list[str]) -> list[str]:
        """Deduplicate similar actions."""
        if not actions:
            return []

        unique = []
        seen = set()
        for action in actions:
            # Normalize for comparison
            key = action.lower().strip()[:50]
            if key not in seen:
                seen.add(key)
                unique.append(action)

        return unique[:10]  # Limit to 10 unique actions per scene

    def _build_action_sequence(self, actions: list[str]) -> str:
        """Build a narrative sequence from actions."""
        if not actions:
            return ""
        if len(actions) == 1:
            return actions[0]

        # Build "First X, then Y, finally Z" narrative
        parts = []
        for i, action in enumerate(actions[:5]):  # Limit to 5 for brevity
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
        """Build a visual summary for embedding."""
        parts = []

        # People
        if person_names:
            parts.append(f"People: {', '.join(person_names)}")

        # Key actions
        if actions:
            parts.append(f"Actions: {'; '.join(actions[:3])}")

        # Key entities
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
    """Convenience function to aggregate frames into scene data.

    Args:
        frames: List of frame data dicts.
        start_time: Scene start timestamp.
        end_time: Scene end timestamp.
        dialogue_segments: ASR segments for the scene.

    Returns:
        Dict ready for SceneData creation.
    """
    aggregator = SceneAggregator()
    return aggregator.aggregate_frames(frames, start_time, end_time, dialogue_segments)
