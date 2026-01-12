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


@dataclass
class Scenelet:
    start_ts: float
    end_ts: float
    frames: list[TemporalContext]
    audio_text: str = ""
    
    @property
    def fused_content(self) -> str:
        frame_descs = " | ".join([f.description for f in self.frames if f.description])
        return f"[{self.start_ts:.1f}s-{self.end_ts:.1f}s] {frame_descs} AUDIO: {self.audio_text}"
    
    @property
    def all_entities(self) -> list[str]:
        entities = []
        for f in self.frames:
            entities.extend(f.entities)
        return list(set(entities))
    
    @property
    def all_actions(self) -> list[str]:
        actions = []
        for f in self.frames:
            actions.extend(f.actions)
        return list(set(actions))


class SceneletBuilder:
    def __init__(self, window_seconds: float = 5.0, stride_seconds: float = 2.5):
        self.window_seconds = window_seconds
        self.stride_seconds = stride_seconds
        self.frames: list[TemporalContext] = []
        self.audio_segments: list[dict] = []
    
    def add_frame(self, ctx: TemporalContext) -> None:
        self.frames.append(ctx)
    
    def set_audio_segments(self, segments: list[dict]) -> None:
        self.audio_segments = segments
    
    def _get_audio_for_window(self, start_ts: float, end_ts: float) -> str:
        texts = []
        for seg in self.audio_segments:
            seg_start = seg.get("start", 0)
            seg_end = seg.get("end", 0)
            if seg_start < end_ts and seg_end > start_ts:
                texts.append(seg.get("text", ""))
        return " ".join(texts)[:200]
    
    def build_scenelets(self) -> list[Scenelet]:
        if not self.frames:
            return []
        
        self.frames.sort(key=lambda f: f.timestamp)
        max_ts = self.frames[-1].timestamp
        scenelets = []
        
        start_ts = 0.0
        while start_ts < max_ts:
            end_ts = start_ts + self.window_seconds
            window_frames = [
                f for f in self.frames 
                if start_ts <= f.timestamp < end_ts
            ]
            if window_frames:
                audio = self._get_audio_for_window(start_ts, end_ts)
                scenelets.append(Scenelet(
                    start_ts=start_ts,
                    end_ts=end_ts,
                    frames=window_frames,
                    audio_text=audio,
                ))
            start_ts += self.stride_seconds
        
        return scenelets
    
    def clear(self) -> None:
        self.frames.clear()
        self.audio_segments.clear()

