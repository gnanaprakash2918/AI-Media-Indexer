"""XMem-style 3-tier temporal context memory for video understanding."""

from collections import deque
from dataclasses import dataclass, field
from typing import Any


@dataclass
class TemporalContext:
    """Represents the semantic state of a single video frame.

    Attributes:
        timestamp: Time in seconds from start of video.
        description: Vision-LLM generated dense caption.
        entities: List of objects/people visible.
        faces: List of face cluster IDs detected.
        actions: List of actions currently in progress.
    """

    timestamp: float
    description: str
    entities: list[str] = field(default_factory=list)
    faces: list[int] = field(default_factory=list)
    actions: list[str] = field(default_factory=list)


class TemporalContextManager:
    """Manages multi-tier temporal memory for video understanding.

    Inspired by XMem, it maintains sensory (short-term), working
    (intermediate), and long-term (identities) state.
    """

    def __init__(self, sensory_size: int = 5, working_size: int = 10) -> None:
        """Initializes the context manager.

        Args:
            sensory_size: Number of previous frames to keep in sliding window.
            working_size: Reserved for intermediate feature storage.
        """
        self.sensory: deque[TemporalContext] = deque(maxlen=sensory_size)
        self.working: dict[str, Any] = {}
        self.long_term: dict[int, str] = {}

    def add_frame(self, ctx: TemporalContext) -> None:
        """Adds a new frame context and updates the active entity map.

        Args:
            ctx: The TemporalContext object for the new frame.
        """
        self.sensory.append(ctx)
        for entity in ctx.entities:
            self.working[entity] = {
                "last_seen": ctx.timestamp,
                "count": self.working.get(entity, {}).get("count", 0) + 1,
            }

    def register_identity(self, cluster_id: int, name: str) -> None:
        """Registers a face cluster ID to a known name in long-term memory.

        Args:
            cluster_id: The database ID of the face cluster.
            name: The human-readable name of the person.
        """
        self.long_term[cluster_id] = name

    def get_context_for_vlm(self) -> str:
        """Constructs a prompt snippet showing recent narrative history.

        Returns:
            A string chain of recent descriptions for grounding the Vision LLM.
        """
        if not self.sensory:
            return "Start of video"
        parts = [
            f"[{c.timestamp:.1f}s] {c.description[:100]}" for c in self.sensory
        ]
        return " -> ".join(parts)

    def get_active_entities(self) -> list[str]:
        """Retrieves a list of entities visible in the most recent sensory frame.

        Returns:
            A list of entity labels.
        """
        return self.sensory[-1].entities if self.sensory else []

    def get_identity_name(self, cluster_id: int) -> str | None:
        """Retrieves the known name for a face cluster ID.

        Args:
            cluster_id: The cluster ID to look up.

        Returns:
            The registered name, or None if unknown.
        """
        return self.long_term.get(cluster_id)

    def clear_sensory(self) -> None:
        """Clears the short-term sensory memory."""
        self.sensory.clear()

    def get_summary(self) -> dict[str, int]:
        """Provides a statistical summary of the memory state.

        Returns:
            A dictionary containing counts for sensory, working, and long-term memory.
        """
        return {
            "sensory_frames": len(self.sensory),
            "working_entities": len(self.working),
            "long_term_identities": len(self.long_term),
        }


@dataclass
class Scenelet:
    """A cohesive temporal segment containing multiple frames and audio.

    Attributes:
        start_ts: Start time in seconds.
        end_ts: End time in seconds.
        frames: List of TemporalContext objects within this window.
        audio_text: Aggregated transcript for this window.
    """

    start_ts: float
    end_ts: float
    frames: list[TemporalContext]
    audio_text: str = ""

    @property
    def fused_content(self) -> str:
        """Combines visual descriptions and audio text into one narrative string."""
        frame_descs = " | ".join(
            [f.description for f in self.frames if f.description]
        )
        return f"[{self.start_ts:.1f}s-{self.end_ts:.1f}s] {frame_descs} AUDIO: {self.audio_text}"

    @property
    def all_entities(self) -> list[str]:
        """Returns a unique list of all entities visible in this scenelet."""
        entities = []
        for f in self.frames:
            entities.extend(f.entities)
        return list(set(entities))

    @property
    def all_actions(self) -> list[str]:
        """Returns a unique list of all actions occurring in this scenelet."""
        actions = []
        for f in self.frames:
            actions.extend(f.actions)
        return list(set(actions))


class SceneletBuilder:
    """Building scene-level aggregations from frame-level data.

    Uses a sliding window approach to group frames and associated audio.
    """

    def __init__(
        self, window_seconds: float = 5.0, stride_seconds: float = 2.5
    ) -> None:
        """Initializes the scenelet builder.

        Args:
            window_seconds: Duration of each scenelet in seconds.
            stride_seconds: Step size between windows.
        """
        self.window_seconds = window_seconds
        self.stride_seconds = stride_seconds
        self.frames: list[TemporalContext] = []
        self.audio_segments: list[dict[str, Any]] = []

    def add_frame(self, ctx: TemporalContext) -> None:
        """Adds a frame context to the internal buffer for later aggregation."""
        self.frames.append(ctx)

    def set_audio_segments(self, segments: list[dict[str, Any]]) -> None:
        """Provides the transcriber output to be matched against windows."""
        self.audio_segments = segments

    def _get_audio_for_window(self, start_ts: float, end_ts: float) -> str:
        """Extracts and joins all audio segments that overlap with a time window."""
        texts = []
        for seg in self.audio_segments:
            seg_start = seg.get("start", 0)
            seg_end = seg.get("end", 0)
            if seg_start < end_ts and seg_end > start_ts:
                texts.append(seg.get("text", ""))
        return " ".join(texts)[:200]

    def build_scenelets(self) -> list[Scenelet]:
        """Processes buffered frames and audio into a list of fused Scenelets.

        Returns:
            A list of aggregated Scenelet objects.
        """
        if not self.frames:
            return []

        self.frames.sort(key=lambda f: f.timestamp)
        max_ts = self.frames[-1].timestamp
        scenelets = []

        start_ts = 0.0
        while start_ts < max_ts:
            end_ts = start_ts + self.window_seconds
            window_frames = [
                f for f in self.frames if start_ts <= f.timestamp < end_ts
            ]
            if window_frames:
                audio = self._get_audio_for_window(start_ts, end_ts)
                scenelets.append(
                    Scenelet(
                        start_ts=start_ts,
                        end_ts=end_ts,
                        frames=window_frames,
                        audio_text=audio,
                    )
                )
            start_ts += self.stride_seconds

        return scenelets

    def clear(self) -> None:
        """Clears all buffered frame and audio data."""
        self.frames.clear()
        self.audio_segments.clear()
