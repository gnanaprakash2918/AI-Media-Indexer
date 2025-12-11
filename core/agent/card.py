"""A2A AgentCard definition for the Media Indexer agent.

This module exposes a single helper, :func:`get_agent_card`, which
constructs an :class:`AgentCard` describing the capabilities of the
MediaIndexer agent for A2A-compatible orchestrators.
"""

from __future__ import annotations

from a2a.types import AgentCapabilities, AgentCard, AgentSkill, TransportProtocol


def get_agent_card(base_url: str = "http://localhost:8000") -> AgentCard:
    """Build and return the AgentCard for this media agent.

    Args:
        base_url: Base HTTP URL where the A2A server is reachable.

    Returns:
        A fully-populated :class:`AgentCard` instance describing the
        MediaIndexer agent, its skills, and supported transports.
    """
    return AgentCard(
        name="MediaIndexer",
        description=(
            "A multimodal agent that ingests, indexes, and searches "
            "personal and media libraries using audio, subtitles, and "
            "visual embeddings."
        ),
        version="1.0.0",
        url=f"{base_url}/a2a",
        preferred_transport=TransportProtocol.http_json,
        capabilities=AgentCapabilities(
            streaming=True,
            push_notifications=False,
        ),
        default_input_modes=["application/json"],
        default_output_modes=["application/json"],
        skills=[
            AgentSkill(
                id="search_media",
                name="Search Media",
                description=(
                    "Finds relevant media segments via natural language "
                    "queries over dialogue and visual frames."
                ),
                tags=["search", "video", "retrieval"],
            ),
            AgentSkill(
                id="ingest_media",
                name="Ingest Media",
                description=(
                    "Ingests and indexes video files, extracting metadata, "
                    "subtitles/transcripts, visual features, and faces."
                ),
                tags=["ingest", "processing", "indexing"],
            ),
        ],
    )
