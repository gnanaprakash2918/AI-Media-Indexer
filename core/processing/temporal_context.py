"""Temporal context memory for video understanding.

XMem-style 3-tier memory for maintaining temporal coherence:
- Sensory: Last 3 frames (immediate context)
- Working: Current entities being tracked
- Long-term: Persistent identities across video
"""

from collections import deque
from dataclasses import dataclass, field
from typing import Any


@dataclass
class TemporalContext:
    """Context from a single frame."""
    timestamp: float
    description: str
    entities: list[str] = field(default_factory=list)
    faces: list[int] = field(default_factory=list)
    actions: list[str] = field(default_factory=list)


class TemporalContextManager:
    """3-tier memory for temporal coherence across frames."""

    def __init__(self, sensory_size: int = 3, working_size: int = 10):
        self.sensory: deque[TemporalContext] = deque(maxlen=sensory_size)
        self.working: dict[str, Any] = {}  # entity_name -> tracking data
        self.long_term: dict[int, str] = {}  # cluster_id -> identity name
    
    def add_frame(self, ctx: TemporalContext) -> None:
        """Add frame context to sensory memory."""
        self.sensory.append(ctx)
        
        # Update working memory with entities
        for entity in ctx.entities:
            self.working[entity] = {
                "last_seen": ctx.timestamp,
                "count": self.working.get(entity, {}).get("count", 0) + 1,
            }
        
        # Update face tracking
        for face_id in ctx.faces:
            if face_id in self.long_term:
                continue
            # Keep track of faces for identity resolution

    def register_identity(self, cluster_id: int, name: str) -> None:
        """Register a face cluster with a name (HITL)."""
        self.long_term[cluster_id] = name

    def get_context_for_vlm(self) -> str:
        """Generate context string for VLM prompt."""
        if not self.sensory:
            return "Start of video"
        
        parts = []
        for ctx in self.sensory:
            ts = ctx.timestamp
            desc = ctx.description[:100] if ctx.description else ""
            parts.append(f"[{ts:.1f}s] {desc}")
        
        return " -> ".join(parts)

    def get_active_entities(self) -> list[str]:
        """Get entities visible in recent frames."""
        if not self.sensory:
            return []
        
        recent = self.sensory[-1]
        return recent.entities

    def get_identity_name(self, cluster_id: int) -> str | None:
        """Get HITL identity name for a face cluster."""
        return self.long_term.get(cluster_id)

    def clear_sensory(self) -> None:
        """Clear sensory memory (e.g., on scene change)."""
        self.sensory.clear()

    def get_summary(self) -> dict:
        """Get memory state summary."""
        return {
            "sensory_frames": len(self.sensory),
            "working_entities": len(self.working),
            "long_term_identities": len(self.long_term),
        }
