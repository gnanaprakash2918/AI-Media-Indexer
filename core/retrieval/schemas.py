"""Structured Query Schemas for Agentic Search.

Defines Pydantic models for query decomposition and structured search.
Used by the Query Decoupler to parse complex natural language queries
into structured, searchable components.
"""

from __future__ import annotations

from enum import Enum
from typing import Any

from pydantic import BaseModel, Field


class QueryModality(str, Enum):
    """Modalities supported in search."""
    VISUAL = "visual"
    AUDIO = "audio"
    TEXT = "text"
    IDENTITY = "identity"


class EntityRelationship(BaseModel):
    """Relationship between entities in query."""
    relation: str = Field(description="Relationship type (wearing, holding, near)")
    target: str = Field(description="Target entity of the relationship")


class QueryEntity(BaseModel):
    """Extracted entity from query with attributes and relationships."""
    entity_type: str = Field(description="Type: person, vehicle, clothing, location, etc.")
    name: str = Field(description="Name or label of the entity")
    attributes: dict[str, Any] = Field(default_factory=dict, description="Key-value attributes")
    relationships: list[EntityRelationship] = Field(default_factory=list)


class TemporalConstraint(BaseModel):
    """Time-related constraint from query."""
    type: str = Field(description="Type: before, after, during, slow, fast")
    reference: str | None = Field(default=None, description="Reference point if any")
    value: str | None = Field(default=None, description="Extracted value")


class StructuredQuery(BaseModel):
    """Fully decomposed query for VideoRAG search.
    
    Produced by the Query Decoupler LLM chain from natural language input.
    Consumed by the VideoRAG orchestrator to perform multi-modal search.
    """
    # Raw query for reference
    original_query: str = Field(description="Original user query")

    # Decomposed cues by modality
    visual_cues: list[str] = Field(
        default_factory=list,
        description="Visual descriptions to search for (colors, objects, actions)"
    )
    audio_cues: list[str] = Field(
        default_factory=list,
        description="Audio/dialogue cues to search for (spoken words, sounds)"
    )
    text_cues: list[str] = Field(
        default_factory=list,
        description="On-screen text or entity names to match"
    )

    # Identity Filters (names of people to filter)
    identities: list[str] = Field(
        default_factory=list,
        description="Person names that should appear in results"
    )

    # Extracted entities with full structure
    entities: list[QueryEntity] = Field(
        default_factory=list,
        description="Fully extracted entities with relationships"
    )

    # Temporal constraints
    temporal_cues: list[str] = Field(
        default_factory=list,
        description="Time-related constraints (slowly, before, after)"
    )
    temporal_constraints: list[TemporalConstraint] = Field(
        default_factory=list,
        description="Structured temporal constraints"
    )

    # Flags for advanced processing
    requires_external_knowledge: bool = Field(
        default=False,
        description="True if query needs external web search for context"
    )
    is_question: bool = Field(
        default=False,
        description="True if query is a question requiring an answer"
    )

    # Dense scene description for semantic search
    scene_description: str = Field(
        default="",
        description="Combined dense description for vector search"
    )

    # Which modalities to search
    modalities: list[QueryModality] = Field(
        default_factory=lambda: [QueryModality.VISUAL],
        description="Modalities to include in search"
    )

    # Confidence in decomposition
    decomposition_confidence: float = Field(
        default=1.0,
        ge=0.0,
        le=1.0,
        description="Confidence score for the decomposition"
    )


class SearchResultItem(BaseModel):
    """Single search result with explainability."""
    id: str
    video_path: str
    timestamp: float
    score: float

    # Match explanations
    match_reasons: list[str] = Field(default_factory=list)
    matched_entities: list[str] = Field(default_factory=list)
    matched_constraints: list[str] = Field(default_factory=list)

    # Content
    action: str | None = None
    dialogue: str | None = None
    entities: list[str] = Field(default_factory=list)
    face_names: list[str] = Field(default_factory=list)

    # Scores by component
    visual_score: float | None = None
    audio_score: float | None = None
    identity_score: float | None = None
    rrf_score: float | None = None


class VideoRAGResponse(BaseModel):
    """Response from VideoRAG orchestrator.
    
    Includes both retrieved clips and generated answer (if question).
    """
    # The structured query used
    query: StructuredQuery

    # Retrieved results
    results: list[SearchResultItem] = Field(default_factory=list)
    total_results: int = 0

    # Generated answer (if is_question=True)
    answer: str | None = None
    answer_citations: list[str] = Field(default_factory=list)
    answer_confidence: float = 0.0

    # External enrichment (if requires_external_knowledge=True)
    external_context: dict[str, Any] = Field(default_factory=dict)

    # Timing
    decomposition_time_ms: float = 0.0
    search_time_ms: float = 0.0
    rerank_time_ms: float = 0.0
    total_time_ms: float = 0.0


class EnrichmentResult(BaseModel):
    """Result from external knowledge enrichment."""
    entity_name: str
    possible_matches: list[str] = Field(default_factory=list)
    confidence: float = 0.0
    source: str = "brave_search"
    context: str = ""
