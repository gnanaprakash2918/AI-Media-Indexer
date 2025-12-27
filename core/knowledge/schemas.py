"""Universal structured knowledge schemas for FAANG-level frame analysis.

This module uses Pydantic schemas to enforce structured output from the LLM,
WITHOUT hardcoded lists. The LLM's internal world knowledge handles specificity.
"""

from __future__ import annotations

from pydantic import BaseModel, Field, field_validator


class EntityDetail(BaseModel):
    """A specific entity detected in the frame."""

    name: str = Field(
        ...,
        description="Specific name (e.g., 'Idly', 'Nike Air Jordan', 'Katana', 'Tesla Model 3')",
    )
    category: str = Field(
        ...,
        description="Category (e.g., 'Food', 'Footwear', 'Weapon', 'Vehicle')",
    )
    visual_details: str = Field(
        default="",
        description="Color, texture, state (e.g., 'Steaming hot', 'Worn out', 'Bright red')",
    )
    
    @field_validator('visual_details', mode='before')
    @classmethod
    def convert_dict_to_string(cls, v):
        """Convert dict visual_details to string (Ollama sometimes returns dict)."""
        if isinstance(v, dict):
            # Convert {'color': 'Blue', 'pattern': 'Checkered'} to "Blue, Checkered"
            return ", ".join(str(val) for val in v.values() if val)
        return v or ""


class SceneContext(BaseModel):
    """Context about the scene/location."""

    location: str = Field(
        default="",
        description="Specific location (e.g., 'Bowling Alley', 'Temple', 'Office')",
    )
    action_narrative: str = Field(
        default="",
        description="Physical action (e.g., 'Pin wobbling before falling', 'Dipping idli in sambar')",
    )
    cultural_context: str | None = Field(
        default=None,
        description="Cultural setting if distinct (e.g., 'South Indian Wedding', 'Japanese Tea Ceremony')",
    )
    visible_text: list[str] = Field(
        default_factory=list,
        description="All readable text/brands in the scene",
    )
    time_of_day: str = Field(
        default="",
        description="Time indicator from lighting/context (morning/afternoon/evening/night)",
    )


class FrameAnalysis(BaseModel):
    """Universal structured knowledge for any video frame.

    This schema is culturally agnostic - the LLM's world knowledge
    provides specificity (Idly vs rice cake, Saree vs dress).
    """

    # Core content
    main_subject: str = Field(
        default="",
        description="Main person or object in focus",
    )
    action: str = Field(
        default="",
        description="The precise action occurring (e.g., 'bowling a strike', 'eating breakfast')",
    )
    action_physics: str = Field(
        default="",
        description="Physical details (e.g., 'ball spinning fast', 'pin falling slowly')",
    )

    # Entities (food, objects, clothing - all in one universal list)
    entities: list[EntityDetail] = Field(
        default_factory=list,
        description="Key objects, foods, clothing, or tools with specificity",
    )

    # Scene context
    scene: SceneContext = Field(
        default_factory=SceneContext,
        description="Scene/location information",
    )

    # Identity linkage (filled by pipeline, not LLM)
    face_ids: list[str] = Field(
        default_factory=list,
        description="Face cluster IDs detected in this frame",
    )
    person_names: list[str] = Field(
        default_factory=list,
        description="Names resolved from HITL naming",
    )

    # Audio linkage
    speaker_ids: list[str] = Field(
        default_factory=list,
        description="Voice segment IDs overlapping this timestamp",
    )

    def to_search_content(self) -> str:
        """Generate rich searchable text for embedding.

        This creates a dense text representation optimized for
        semantic search, with priority ordering.
        """
        parts: list[str] = []

        # 1. IDENTITY (highest weight - enables "Prakash bowling")
        for name in self.person_names:
            parts.extend([name, name])  # Double for emphasis

        # 2. ACTION (second priority)
        if self.action:
            parts.append(self.action)
        if self.action_physics:
            parts.append(self.action_physics)
        if self.main_subject:
            parts.append(self.main_subject)

        # 3. ENTITIES (food, clothing, objects with details)
        for entity in self.entities:
            entity_str = f"{entity.name}"
            if entity.visual_details:
                entity_str += f" ({entity.visual_details})"
            parts.append(entity_str)

        # 4. VISIBLE TEXT (brands, signs)
        parts.extend(self.scene.visible_text)

        # 5. SCENE
        if self.scene.location:
            parts.append(self.scene.location)
        if self.scene.cultural_context:
            parts.append(self.scene.cultural_context)
        if self.scene.action_narrative:
            parts.append(self.scene.action_narrative)

        return " ".join(filter(None, parts))

    def to_payload(self) -> dict:
        """Convert to Qdrant-storable payload with indexed fields."""
        return {
            "main_subject": self.main_subject,
            "action": self.action,
            "action_physics": self.action_physics,
            "entities": [e.model_dump() for e in self.entities],
            "entity_names": [e.name for e in self.entities],  # For filtering
            "entity_categories": list({e.category for e in self.entities}),
            "scene_location": self.scene.location,
            "scene_cultural": self.scene.cultural_context or "",
            "scene_narrative": self.scene.action_narrative,
            "visible_text": self.scene.visible_text,
            "time_of_day": self.scene.time_of_day,
            "face_ids": self.face_ids,
            "person_names": self.person_names,
            "speaker_ids": self.speaker_ids,
        }


class ParsedQuery(BaseModel):
    """Structured representation of a user search query.

    Generated by LLM to enable intelligent search expansion.
    """

    person_name: str | None = Field(
        default=None,
        description="Person name if mentioned (e.g., 'Prakash', 'Mom')",
    )
    visual_keywords: list[str] = Field(
        default_factory=list,
        description="Expanded visual keywords (e.g., 'South Indian breakfast' -> ['idli', 'dosa', 'sambar'])",
    )
    action_keywords: list[str] = Field(
        default_factory=list,
        description="Action keywords (e.g., 'bowling', 'eating', 'running')",
    )
    text_to_find: list[str] = Field(
        default_factory=list,
        description="Specific text/brands to search (e.g., 'Nike', 'Brunswick')",
    )
    temporal_hints: list[str] = Field(
        default_factory=list,
        description="Time-related hints (e.g., 'slowly', 'at the end')",
    )

    def to_search_text(self) -> str:
        """Generate search text from parsed components."""
        parts = []
        if self.person_name:
            parts.append(self.person_name)
        parts.extend(self.visual_keywords)
        parts.extend(self.action_keywords)
        parts.extend(self.text_to_find)
        return " ".join(parts)
