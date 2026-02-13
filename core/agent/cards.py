"""A2A Agent Cards for Multi-Agent Orchestration.

Defines capability cards for VisionAgent, AudioAgent, and SearchAgent.
The Orchestrator uses these cards for LLM-based tool routing.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

try:
    from a2a.types import AgentCapabilities, AgentCard, AgentSkill

    HAS_A2A = True
except ImportError:
    HAS_A2A = False


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


def get_vision_agent_card(
    base_url: str = "http://localhost:8000",
) -> AgentCard:
    """Returns the AgentCard for the VisionAgent.

    Args:
        base_url: The base URL where the agent is reachable.

    Returns:
        The vision agent card.
    """
    if not HAS_A2A:
        raise ImportError("a2a package not installed")
    return AgentCard(
        name="VisionAgent",
        description="Analyzes video frames for objects, people, actions, and scene context using VLM and SAM3",
        version="1.0.0",
        url=f"{base_url}/a2a/vision",
        capabilities=AgentCapabilities(
            streaming=False, push_notifications=False
        ),
        default_input_modes=["image/jpeg", "video/mp4"],
        default_output_modes=["application/json"],
        skills=[
            AgentSkill(
                id="analyze_frame",
                name="Analyze Frame",
                description="Dense frame analysis",
                tags=["vision"],
            ),
            AgentSkill(
                id="track_concept",
                name="Track Concept",
                description="SAM3 tracking",
                tags=["segmentation"],
            ),
            AgentSkill(
                id="detect_faces",
                name="Detect Faces",
                description="InsightFace 512D",
                tags=["identity"],
            ),
        ],
    )


def get_audio_agent_card(
    base_url: str = "http://localhost:8000",
) -> AgentCard:
    """Returns the AgentCard for the AudioAgent.

    Args:
        base_url: The base URL where the agent is reachable.

    Returns:
        The audio agent card.
    """
    if not HAS_A2A:
        raise ImportError("a2a package not installed")
    return AgentCard(
        name="AudioAgent",
        description="Transcribes audio using Whisper and performs speaker diarization",
        version="1.0.0",
        url=f"{base_url}/a2a/audio",
        capabilities=AgentCapabilities(
            streaming=True, push_notifications=False
        ),
        default_input_modes=["audio/wav", "video/mp4"],
        default_output_modes=["application/json", "text/srt"],
        skills=[
            AgentSkill(
                id="transcribe",
                name="Transcribe",
                description="ASR with Whisper",
                tags=["audio"],
            ),
            AgentSkill(
                id="diarize",
                name="Diarize Speakers",
                description="Speaker identification",
                tags=["audio"],
            ),
        ],
    )


def get_search_agent_card(
    base_url: str = "http://localhost:8000",
) -> AgentCard:
    """Returns the AgentCard for the SearchAgent.

    Args:
        base_url: The base URL where the agent is reachable.

    Returns:
        The search agent card.
    """
    if not HAS_A2A:
        raise ImportError("a2a package not installed")
    return AgentCard(
        name="SearchAgent",
        description="Agentic search with LLM query expansion, identity resolution, and constraint verification",
        version="1.0.0",
        url=f"{base_url}/a2a/search",
        capabilities=AgentCapabilities(
            streaming=False, push_notifications=False
        ),
        default_input_modes=["application/json"],
        default_output_modes=["application/json"],
        skills=[
            AgentSkill(
                id="search_scenes",
                name="Search Scenes",
                description="Complex visual search",
                tags=["search"],
            ),
            AgentSkill(
                id="search_dialogue",
                name="Search Dialogue",
                description="Transcript search",
                tags=["search"],
            ),
        ],
    )


def get_all_tool_schemas() -> list[dict]:
    """Get all agent tools as JSON schemas for LLM function calling."""
    all_tools = VISION_AGENT_TOOLS + AUDIO_AGENT_TOOLS + SEARCH_AGENT_TOOLS
    return [t.to_json_schema() for t in all_tools]
