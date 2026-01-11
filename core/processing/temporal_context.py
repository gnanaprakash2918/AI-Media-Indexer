"""XMem-style 3-tier temporal context memory for video understanding."""

from collections import deque
from dataclasses import dataclass, field
from typing import Any


@dataclass
class TemporalContext:
    timestamp: float
    description: str
    entities: list[str] = field(default_factory=list)
    faces: list[int] = field(default_factory=list)
    actions: list[str] = field(default_factory=list)


class TemporalContextManager:
    """3-tier memory: sensory (recent), working (tracking), long-term (identities)."""

    def __init__(self, sensory_size: int = 5, working_size: int = 10):
        self.sensory: deque[TemporalContext] = deque(maxlen=sensory_size)
        self.working: dict[str, Any] = {}
        self.long_term: dict[int, str] = {}

    def add_frame(self, ctx: TemporalContext) -> None:
        self.sensory.append(ctx)
        for entity in ctx.entities:
            self.working[entity] = {
                "last_seen": ctx.timestamp,
                "count": self.working.get(entity, {}).get("count", 0) + 1,
            }

    def register_identity(self, cluster_id: int, name: str) -> None:
        self.long_term[cluster_id] = name

    def get_context_for_vlm(self) -> str:
        if not self.sensory:
            return "Start of video"
        parts = [f"[{c.timestamp:.1f}s] {c.description[:100]}" for c in self.sensory]
        return " -> ".join(parts)

    def get_active_entities(self) -> list[str]:
        return self.sensory[-1].entities if self.sensory else []

    def get_identity_name(self, cluster_id: int) -> str | None:
        return self.long_term.get(cluster_id)

    def clear_sensory(self) -> None:
        self.sensory.clear()

    def get_summary(self) -> dict:
        return {
            "sensory_frames": len(self.sensory),
            "working_entities": len(self.working),
            "long_term_identities": len(self.long_term),
        }
