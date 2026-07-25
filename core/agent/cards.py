"""A2A Agent Cards for Multi-Agent Orchestration.

Defines capability cards for VisionAgent, AudioAgent, and SearchAgent.
The Orchestrator uses these cards for LLM-based tool routing.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class ToolParameter:
    """Parameter definition for an agent tool."""

    name: str
    type: str
    description: str
    required: bool = True


@dataclass
class AgentToolCard:
    """JSON-Schema compatible tool definition for LLM routing."""

    name: str
    description: str
    parameters: list[ToolParameter] = field(default_factory=list)

    def to_json_schema(self) -> dict[str, Any]:
        """Converts the tool card to a JSON-Schema compatible dictionary."""
        props = {}
        required = []
        for p in self.parameters:
            props[p.name] = {"type": p.type, "description": p.description}
            if p.required:
                required.append(p.name)
        return {
            "name": self.name,
            "description": self.description,
            "parameters": {
                "type": "object",
                "properties": props,
                "required": required,
            },
        }


# Vision Agent Tools
VISION_AGENT_TOOLS = [
    AgentToolCard(
        name="analyze_frame",
        description="Analyze a video frame for objects, people, actions, text, and scene context",
        parameters=[
            ToolParameter("frame_path", "string", "Path to the frame image"),
            ToolParameter(
                "temporal_context",
                "string",
                "Previous frame descriptions",
                required=False,
            ),
        ],
    ),
    AgentToolCard(
        name="track_concept",
        description="Track a visual concept (person, object) across video using SAM3",
        parameters=[
            ToolParameter("video_path", "string", "Path to video file"),
            ToolParameter(
                "concept",
                "string",
                "Concept to track (e.g., 'red car', 'person')",
            ),
        ],
    ),
    AgentToolCard(
        name="detect_faces",
        description="Detect and recognize faces in a frame using InsightFace 512D embeddings",
        parameters=[
            ToolParameter("frame_path", "string", "Path to the frame image"),
        ],
    ),
]

# Audio Agent Tools
AUDIO_AGENT_TOOLS = [
    AgentToolCard(
        name="transcribe_audio",
        description="Transcribe audio to text using Whisper",
        parameters=[
            ToolParameter("audio_path", "string", "Path to audio/video file"),
            ToolParameter(
                "language",
                "string",
                "Language code (e.g., 'en', 'fr')",
                required=False,
            ),
        ],
    ),
    AgentToolCard(
        name="diarize_speakers",
        description="Identify and separate different speakers in audio",
        parameters=[
            ToolParameter("audio_path", "string", "Path to audio file"),
        ],
    ),
]

# Search Agent Tools
SEARCH_AGENT_TOOLS = [
    AgentToolCard(
        name="search_scenes",
        description="Search for video scenes matching a natural language query with identity and action filtering",
        parameters=[
            ToolParameter("query", "string", "Natural language search query"),
            ToolParameter(
                "person",
                "string",
                "Person name to filter by (HITL-assigned)",
                required=False,
            ),
            ToolParameter(
                "limit", "integer", "Maximum results", required=False
            ),
        ],
    ),
    AgentToolCard(
        name="search_dialogue",
        description="Search transcripts and spoken dialogue",
        parameters=[
            ToolParameter(
                "query", "string", "Text to search for in transcripts"
            ),
        ],
    ),
]


def get_all_tool_schemas() -> list[dict]:
    """Get all agent tools as JSON schemas for LLM function calling."""
    all_tools = VISION_AGENT_TOOLS + AUDIO_AGENT_TOOLS + SEARCH_AGENT_TOOLS
    return [t.to_json_schema() for t in all_tools]


