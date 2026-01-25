"""Identity Linker Engine - Links Face, Voice, and TMDB Cast.

Calculates temporal co-occurrence between Face and Voice clusters,
fuzzy matches against TMDB cast, and generates merge suggestions.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass
class IdentitySuggestion:
    """Suggestion for merging or matching identities."""

    type: (
        str  # "merge_face_voice", "merge_face_face", "tmdb_match", "ner_match"
    )
    source: str
    target: str
    reason: str
    confidence: float
    source_id: int | None = None
    target_id: int | None = None
    strict_mode: bool = (
        True  # If True, requires strict user review (personal ID)
    )

    def to_dict(self) -> dict:
        """Converts the suggestion to a dictionary format."""
        return {
            "type": self.type,
            "source": self.source,
            "target": self.target,
            "reason": self.reason,
            "confidence": round(self.confidence, 2),
            "source_id": self.source_id,
            "target_id": self.target_id,
            "strict_mode": self.strict_mode,
        }


class IdentityLinker:
    """Links Face, Voice, and TMDB Cast identities through temporal analysis."""

    def __init__(
        self, overlap_threshold: float = 0.8, fuzzy_threshold: float = 0.75
    ):
        """Initializes the IdentityLinker.

        Args:
            overlap_threshold: Minimum temporal overlap ratio for suggesting links.
            fuzzy_threshold: Minimum name similarity for suggesting links.
        """
        self.overlap_threshold = overlap_threshold
        self.fuzzy_threshold = fuzzy_threshold

    def calculate_temporal_overlap(
        self,
        timestamps_a: list[tuple[float, float]],
        timestamps_b: list[tuple[float, float]],
    ) -> float:
        """Calculate overlap ratio between two sets of time ranges."""
        if not timestamps_a or not timestamps_b:
            return 0.0

        total_a = sum(end - start for start, end in timestamps_a)
        if total_a == 0:
            return 0.0

        overlap_time = 0.0
        for a_start, a_end in timestamps_a:
            for b_start, b_end in timestamps_b:
                overlap_start = max(a_start, b_start)
                overlap_end = min(a_end, b_end)
                if overlap_end > overlap_start:
                    overlap_time += overlap_end - overlap_start

        return min(overlap_time / total_a, 1.0)

    def fuzzy_match(self, name1: str, name2: str) -> float:
        """Fuzzy name matching using token overlap (Jaccard similarity)."""

        def normalize(s: str) -> set[str]:
            s = re.sub(r"[^\w\s]", "", s.lower())
            return set(s.split())

        tokens1 = normalize(name1)
        tokens2 = normalize(name2)

        if not tokens1 or not tokens2:
            return 0.0

        intersection = tokens1 & tokens2
        union = tokens1 | tokens2

        return len(intersection) / len(union)

    def suggest_face_voice_links(
        self,
        face_clusters: list[dict],
        voice_clusters: list[dict],
    ) -> list[IdentitySuggestion]:
        """Suggest links between face and voice clusters based on temporal overlap."""
        suggestions = []

        for face in face_clusters:
            face_id = face.get("cluster_id")
            face_name = face.get("name") or f"Face #{face_id}"
            face_times = face.get("timestamps", [])

            if not face_times:
                continue

            face_ranges = [
                (t - 1.0, t + 1.0)
                for t in face_times
                if isinstance(t, (int, float))
            ]

            for voice in voice_clusters:
                voice_id = voice.get("cluster_id")
                voice_name = voice.get("name") or f"Voice #{voice_id}"
                voice_times = voice.get("timestamps", [])

                if not voice_times:
                    continue

                voice_ranges = []
                for segment in voice_times:
                    if isinstance(segment, dict):
                        voice_ranges.append(
                            (segment.get("start", 0), segment.get("end", 0))
                        )
                    elif isinstance(segment, (int, float)):
                        voice_ranges.append((segment - 1.0, segment + 1.0))

                overlap = self.calculate_temporal_overlap(
                    face_ranges, voice_ranges
                )

                if overlap >= self.overlap_threshold:
                    suggestions.append(
                        IdentitySuggestion(
                            type="merge_face_voice",
                            source=face_name,
                            target=voice_name,
                            reason=f"High temporal overlap ({overlap:.0%})",
                            confidence=overlap,
                            source_id=face_id,
                            target_id=voice_id,
                            strict_mode=True,  # Linking face to voice is a personal identity action
                        )
                    )

        suggestions.sort(key=lambda x: -x.confidence)
        return suggestions[:10]

    def suggest_tmdb_matches(
        self,
        clusters: list[dict],
        cast_list: list[dict],
    ) -> list[IdentitySuggestion]:
        """Suggest matches between user-labeled clusters and TMDB cast."""
        suggestions = []

        for cluster in clusters:
            cluster_id = cluster.get("cluster_id")
            cluster_name = (cluster.get("name") or "").strip()

            if not cluster_name or cluster_name.lower() in (
                "unknown",
                "unnamed",
            ):
                continue

            for cast in cast_list:
                cast_name = cast.get("name", "")
                character = cast.get("character", "")

                score = self.fuzzy_match(cluster_name, cast_name)
                char_score = self.fuzzy_match(cluster_name, character)
                best_score = max(score, char_score)

                if best_score >= self.fuzzy_threshold:
                    suggestions.append(
                        IdentitySuggestion(
                            type="tmdb_match",
                            source=f"Cluster '{cluster_name}'",
                            target=f"{cast_name} as {character}",
                            reason=f"Fuzzy match ({best_score:.0%})",
                            confidence=best_score,
                            source_id=cluster_id,
                            strict_mode=False,  # Public data match, less strict review needed
                        )
                    )

        suggestions.sort(key=lambda x: -x.confidence)
        return suggestions[:10]

    def suggest_ner_matches(
        self,
        face_clusters: list[dict],
        entity_co_occurrences: dict[int, dict[str, int]],
    ) -> list[IdentitySuggestion]:
        """Suggest names based on frequent Entity (NER) co-occurrence.

        Args:
            face_clusters: List of face cluster info.
            entity_co_occurrences: Dict mapping cluster_id -> {entity_name: count}.
        """
        suggestions = []

        for cluster in face_clusters:
            cid = cluster.get("cluster_id")
            if cid not in entity_co_occurrences:
                continue

            entities = entity_co_occurrences[cid]
            if not entities:
                continue

            # Find top entity
            top_entity, count = max(entities.items(), key=lambda x: x[1])
            total_sightings = len(cluster.get("timestamps", []))

            # Heuristic: If entity appears in >30% of frames for this person
            confidence = min(
                count / max(total_sightings, 1) * 2.0, 1.0
            )  # Boosted confidence

            if confidence > 0.4:
                suggestions.append(
                    IdentitySuggestion(
                        type="ner_match",
                        source=f"Face #{cid}",
                        target=top_entity,
                        reason=f"Entity '{top_entity}' co-occurs frequently ({count} times)",
                        confidence=confidence,
                        source_id=cid,
                        strict_mode=False,  # NER/Celeb is public info
                    )
                )

        suggestions.sort(key=lambda x: -x.confidence)
        return suggestions[:10]

    def suggest_face_merges(
        self,
        face_clusters: list[dict],
    ) -> list[IdentitySuggestion]:
        """Suggest face cluster merges based on name similarity."""
        suggestions = []
        seen = set()

        for i, c1 in enumerate(face_clusters):
            name1 = (c1.get("name") or "").strip().lower()
            if not name1:
                continue

            for j, c2 in enumerate(face_clusters):
                if i >= j:
                    continue

                name2 = (c2.get("name") or "").strip().lower()
                if not name2:
                    continue

                pair = tuple(
                    sorted(
                        [str(c1.get("cluster_id")), str(c2.get("cluster_id"))]
                    )
                )
                if pair in seen:
                    continue
                seen.add(pair)

                score = self.fuzzy_match(name1, name2)
                if score >= self.fuzzy_threshold:
                    suggestions.append(
                        IdentitySuggestion(
                            type="merge_face_face",
                            source=f"Face #{c1.get('cluster_id')} ({name1})",
                            target=f"Face #{c2.get('cluster_id')} ({name2})",
                            reason=f"Similar names ({score:.0%})",
                            confidence=score,
                            source_id=c1.get("cluster_id"),
                            target_id=c2.get("cluster_id"),
                            strict_mode=True,  # Merging faces is strict
                        )
                    )

        return suggestions[:5]

    def get_all_suggestions(
        self,
        face_clusters: list[dict],
        voice_clusters: list[dict],
        tmdb_cast: list[dict] | None = None,
        entity_occurrences: dict[int, dict[str, int]] | None = None,
    ) -> list[dict]:
        """Get all identity suggestions across all types."""

        all_suggestions = []

        all_suggestions.extend(
            self.suggest_face_voice_links(face_clusters, voice_clusters)
        )
        all_suggestions.extend(self.suggest_face_merges(face_clusters))

        if tmdb_cast:
            all_suggestions.extend(
                self.suggest_tmdb_matches(face_clusters, tmdb_cast)
            )

        if entity_occurrences:
            all_suggestions.extend(
                self.suggest_ner_matches(face_clusters, entity_occurrences)
            )

        all_suggestions.sort(key=lambda x: -x.confidence)

        return [s.to_dict() for s in all_suggestions[:15]]


_linker: IdentityLinker | None = None


def get_identity_linker() -> IdentityLinker:
    """Retrieves the singleton instance of the IdentityLinker.

    Returns:
        The initialized IdentityLinker instance.
    """
    global _linker
    if _linker is None:
        _linker = IdentityLinker()
    return _linker
