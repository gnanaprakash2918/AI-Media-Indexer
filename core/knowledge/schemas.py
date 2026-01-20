"""Universal structured knowledge schemas for FAANG-level frame analysis.

This module uses Pydantic schemas to enforce structured output from the LLM,
WITHOUT hardcoded lists. The LLM's internal world knowledge handles specificity.
"""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field, field_validator

# =============================================================================
# DYNAMIC ENTITY SYSTEM (NO HARDCODING - UNLIMITED COMPLEXITY)
# =============================================================================


class DynamicEntity(BaseModel):
    """Fully flexible entity that can represent ANYTHING in a video.

    This replaces hardcoded classes like ClothingItem, VehicleInQuery, etc.
    The LLM dynamically determines the entity type and attributes.

    Examples:
    - {entity_type: "person", name: "Prakash", attributes: {position: "center"}}
    - {entity_type: "vehicle", name: "Ferrari", attributes: {color: "red", model: "488"}}
    - {entity_type: "clothing", name: "t-shirt", attributes: {color: "blue", body_part: "upper"}}
    - {entity_type: "sound", name: "engine roar", attributes: {intensity: "loud"}}
    - {entity_type: "text", name: "Brunswick", attributes: {location: "sign"}}
    - {entity_type: "action", name: "bowling", attributes: {result: "strike"}}
    - {entity_type: "emotion", name: "excitement", attributes: {intensity: "high"}}
    """

    entity_type: str = Field(
        ...,
        description="LLM-determined type: person/vehicle/clothing/accessory/brand/"
        "object/action/sound/text/emotion/location/food/animal/etc",
    )
    name: str = Field(..., description="Entity name or identifier")
    attributes: dict[str, Any] = Field(
        default_factory=dict,
        description="Dynamic key-value attributes - NO FIXED SCHEMA. "
        "Can include: color, size, brand, position, state, material, "
        "body_part, intensity, speed, direction, model, etc.",
    )
    relationships: list[dict[str, str]] = Field(
        default_factory=list,
        description="Relations to other entities: "
        "[{relation: 'wears', target: 'blue shirt'}, "
        "{relation: 'driving', target: 'red ferrari'}]",
    )
    confidence: float = Field(
        default=1.0, description="Detection confidence 0.0-1.0"
    )
    temporal_info: dict[str, Any] = Field(
        default_factory=dict,
        description="Time-related info: {appears_at: 5.2, duration: 3.1, sequence: 'first'}",
    )
    spatial_info: dict[str, Any] = Field(
        default_factory=dict,
        description="Position info: {position: 'left', bbox: [x,y,w,h], depth: 'foreground'}",
    )

    def to_search_text(self) -> str:
        """Generates a dense searchable text representation of the entity.

        Combines entity type, name, attributes, and relationships into a
        single string optimized for vector similarity search.

        Returns:
            A space-separated string of descriptive keywords.
        """
        parts = [self.entity_type, self.name]
        for _key, val in self.attributes.items():
            if val:
                parts.append(f"{val}")
        for rel in self.relationships:
            if rel.get("target"):
                parts.append(f"{rel.get('relation', '')} {rel['target']}")
        return " ".join(filter(None, parts))


class DynamicParsedQuery(BaseModel):
    """Unlimited complexity query parser - extracts EVERYTHING dynamically.

    This is the production-grade query parser that handles queries like:
    - "Prakash wearing blue t-shirt with John Jacobs spectacles and red shoe
       on left foot with green shoe on right foot playing ten-pin bowling at
       Brunswick sports hitting a strike where all pins fell down fastly"
    - "red ferrari and yellow lamborghini hurricane passing red signal while
       a guy with nike blue shoes, vr headset, balenciaga bag gets into accident"
    - "scene where someone says 'I love you' while rain is falling and sad music plays"

    NO PREDEFINED CATEGORIES - the LLM determines entity types dynamically.
    """

    # All entities extracted from the query
    entities: list[DynamicEntity] = Field(
        default_factory=list,
        description="ALL entities from query - unlimited types and attributes",
    )

    # Entity relationships as graph edges
    relationships: list[dict[str, Any]] = Field(
        default_factory=list,
        description="Entity-entity relationships: "
        "[{source: 'Prakash', relation: 'wearing', target: 'blue shirt'}]",
    )

    # LLM-generated dense search text
    scene_description: str = Field(
        default="",
        description="LLM-generated comprehensive scene description for semantic search",
    )

    # Temporal constraints
    temporal_constraints: list[str] = Field(
        default_factory=list,
        description="Time constraints: ['at the end', 'takes a moment', 'slowly']",
    )

    # Audio/dialogue constraints
    audio_constraints: list[str] = Field(
        default_factory=list,
        description="Audio requirements: ['says I love you', 'sad music', 'engine sound']",
    )

    # Music structure section (for queries like "during the chorus")
    music_section: str | None = Field(
        default=None,
        description="Music structure section: chorus, verse, bridge, intro, outro, drop, breakdown",
    )

    # High energy filter (for "climax", "peak moment")
    high_energy: bool = Field(
        default=False,
        description="Whether to filter for high-energy moments in the video",
    )

    # The original query
    raw_query: str = Field(default="")

    # Search modality hints
    modalities: list[str] = Field(
        default_factory=list,
        description="Which modalities to search: ['visual', 'audio', 'dialogue', 'text']",
    )

    def to_search_text(self) -> str:
        """Aggregates all query components into a single dense search string.

        Combines entity details, scene description, temporal hints, and audio
        constraints to create a comprehensive query for semantic search.

        Returns:
            A consolidated search string.
        """
        parts = []

        # All entity search texts
        for entity in self.entities:
            parts.append(entity.to_search_text())

        # Scene description
        if self.scene_description:
            parts.append(self.scene_description)

        # Temporal hints
        parts.extend(self.temporal_constraints)

        # Audio hints
        parts.extend(self.audio_constraints)

        return " ".join(filter(None, parts))

    def get_entities_by_type(self, entity_type: str) -> list[DynamicEntity]:
        """Filters extracted entities by their dynamic type.

        Args:
            entity_type: The type string to filter by (e.g., 'person', 'car').

        Returns:
            A list of matching DynamicEntity objects.
        """
        return [
            e
            for e in self.entities
            if e.entity_type.lower() == entity_type.lower()
        ]

    def get_person_names(self) -> list[str]:
        """Extracts all recognized person names from the entities.

        Returns:
            A list of names suitable for identity-based filtering.
        """
        return [
            e.name for e in self.entities if e.entity_type.lower() == "person"
        ]


# =============================================================================
# LEGACY SCHEMAS (kept for backwards compatibility)
# =============================================================================


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

    @field_validator("visual_details", mode="before")
    @classmethod
    def convert_dict_to_string(cls, v):
        """Convert dict/list visual_details to string (Ollama sometimes returns structured data)."""
        if isinstance(v, dict):
            # Convert {'color': 'Blue', 'pattern': 'Checkered'} to "Blue, Checkered"
            return ", ".join(str(val) for val in v.values() if val)
        if isinstance(v, list):
            # Convert ['Blue', 'Checkered'] to "Blue, Checkered"
            return ", ".join(str(val) for val in v if val)
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

    # Validators to handle Ollama returning lists instead of strings
    @field_validator("main_subject", "action", "action_physics", mode="before")
    @classmethod
    def coerce_list_to_string(cls, v):
        """Convert list to string (Ollama sometimes returns lists)."""
        if isinstance(v, list):
            # Join list items: ['Couple'] -> 'Couple', ['A', 'B'] -> 'A, B'
            return ", ".join(str(item) for item in v if item)
        if v is None:
            return ""
        return str(v)

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
        """Generates rich searchable text for vector embedding.

        Creates a dense text representation optimized for semantic search,
        with priority weighting (Identity > Action > Entities > Scene).

        Returns:
            A priority-weighted search string.
        """
        parts: list[str] = []

        # 1. IDENTITY (highest weight - enables "Prakash bowling")
        for name in self.person_names:
            parts.append(name)

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


class PersonAttribute(BaseModel):
    """Attributes of a person in a scene (clothing, accessories)."""

    face_id: str = Field(default="", description="Face cluster ID")
    name: str | None = Field(
        default=None, description="Resolved name from HITL"
    )
    clothing_color: str = Field(
        default="", description="Primary clothing color"
    )
    clothing_type: str = Field(
        default="", description="Clothing type (shirt, saree, etc.)"
    )
    accessories: list[str] = Field(
        default_factory=list, description="Glasses, watch, jewelry, etc."
    )
    position: str = Field(default="", description="left/center/right in frame")


class SceneData(BaseModel):
    """Scene-level aggregated data for production video search.

    Unlike FrameAnalysis (per-frame), SceneData represents a coherent
    sequence with start/end timestamps - the unit of storage/retrieval.
    """

    # Temporal bounds
    start_time: float = Field(
        ..., description="Scene start timestamp in seconds"
    )
    end_time: float = Field(..., description="Scene end timestamp in seconds")
    duration: float = Field(
        default=0.0, description="Scene duration in seconds"
    )

    # Aggregated content from all frames in scene
    visual_summary: str = Field(
        default="",
        description="LLM-generated summary of what happens visually",
    )
    actions: list[str] = Field(
        default_factory=list,
        description="All actions detected (deduplicated)",
    )
    action_sequence: str = Field(
        default="",
        description="Temporal narrative: 'First X, then Y, finally Z'",
    )

    # Entities aggregated from all frames
    entities: list[EntityDetail] = Field(
        default_factory=list,
        description="All entities seen in scene (deduplicated)",
    )
    visible_text: list[str] = Field(
        default_factory=list,
        description="All OCR text from scene",
    )

    # Identity - people visible in this scene
    face_cluster_ids: list[int] = Field(
        default_factory=list,
        description="All face clusters appearing in scene",
    )
    person_names: list[str] = Field(
        default_factory=list,
        description="Named identities in scene",
    )
    person_attributes: list[PersonAttribute] = Field(
        default_factory=list,
        description="Per-person clothing/accessories",
    )

    # Audio for this time range
    speaker_ids: list[int] = Field(
        default_factory=list,
        description="Voice cluster IDs speaking in scene",
    )
    speaker_names: list[str] = Field(
        default_factory=list,
        description="Named speakers in scene",
    )
    dialogue_transcript: str = Field(
        default="",
        description="Full transcript for scene time range",
    )

    # Scene context
    location: str = Field(default="", description="Where the scene takes place")
    cultural_context: str = Field(
        default="", description="Cultural setting if relevant"
    )
    time_of_day: str = Field(
        default="", description="Morning/afternoon/evening/night"
    )

    # Source
    media_path: str = Field(default="", description="Path to source video")
    frame_count: int = Field(default=0, description="Number of frames in scene")
    thumbnail_path: str = Field(
        default="", description="Representative frame thumbnail"
    )

    def to_search_content(self) -> str:
        """Aggregates scene-level data into a single searchable string.

        Combines person attributes, actions, dialogue, and scene context
        into a dense representation for cross-modal retrieval.

        Returns:
            A weighted search string for the entire scene.
        """
        parts: list[str] = []

        # 1. IDENTITY (highest weight)
        for name in self.person_names:
            parts.append(name)
        for name in self.speaker_names:
            if name not in self.person_names:
                parts.append(name)

        # 2. CLOTHING (critical for complex queries)
        for attr in self.person_attributes:
            if attr.name:
                parts.append(attr.name)
            if attr.clothing_color and attr.clothing_type:
                parts.append(f"{attr.clothing_color} {attr.clothing_type}")
            elif attr.clothing_color:
                parts.append(attr.clothing_color)
            parts.extend(attr.accessories)

        # 3. ACTIONS (temporal sequence)
        parts.extend(self.actions)
        if self.action_sequence:
            parts.append(self.action_sequence)

        # 4. DIALOGUE
        if self.dialogue_transcript:
            # Truncate for embedding but keep searchable
            parts.append(self.dialogue_transcript[:500])

        # 5. ENTITIES
        for entity in self.entities:
            entity_str = entity.name
            if entity.visual_details:
                entity_str += f" ({entity.visual_details})"
            parts.append(entity_str)

        # 6. TEXT/BRANDS
        parts.extend(self.visible_text)

        # 7. SCENE
        if self.location:
            parts.append(self.location)
        if self.cultural_context:
            parts.append(self.cultural_context)
        if self.visual_summary:
            parts.append(self.visual_summary)

        return " ".join(filter(None, parts))

    def to_payload(self) -> dict:
        """Convert to Qdrant-storable payload with indexed fields."""
        return {
            # Temporal
            "start_time": self.start_time,
            "end_time": self.end_time,
            "duration": self.duration,
            # Content
            "visual_summary": self.visual_summary,
            "actions": self.actions,
            "action_sequence": self.action_sequence,
            # Entities (indexed for filtering)
            "entity_names": [e.name for e in self.entities],
            "entity_categories": list({e.category for e in self.entities}),
            "visible_text": self.visible_text,
            # Identity (indexed for filtering)
            "face_cluster_ids": self.face_cluster_ids,
            "person_names": self.person_names,
            "clothing_colors": [
                a.clothing_color
                for a in self.person_attributes
                if a.clothing_color
            ],
            "clothing_types": [
                a.clothing_type
                for a in self.person_attributes
                if a.clothing_type
            ],
            "accessories": [
                acc for a in self.person_attributes for acc in a.accessories
            ],
            # Audio
            "speaker_ids": self.speaker_ids,
            "speaker_names": self.speaker_names,
            "dialogue_transcript": self.dialogue_transcript,
            # Scene
            "location": self.location,
            "cultural_context": self.cultural_context,
            "time_of_day": self.time_of_day,
            # Source
            "media_path": self.media_path,
            "frame_count": self.frame_count,
            "thumbnail_path": self.thumbnail_path,
        }


class ClothingItem(BaseModel):
    """A specific clothing item with location and brand details."""

    body_part: str = Field(
        default="",
        description="Body location (e.g., 'upper body', 'left foot', 'right foot', 'head')",
    )
    item_type: str = Field(
        default="",
        description="Type (e.g., 't-shirt', 'shoe', 'spectacles', 'watch')",
    )
    color: str = Field(
        default="", description="Color (e.g., 'blue', 'red and white')"
    )
    brand: str = Field(
        default="",
        description="Brand name if mentioned (e.g., 'Nike', 'John Jacobs', 'Balenciaga')",
    )
    details: str = Field(
        default="",
        description="Additional details (e.g., 'VR headset', 'running shoes')",
    )


class PersonInQuery(BaseModel):
    """A person mentioned in the query with all their attributes."""

    name: str | None = Field(
        default=None,
        description="Person name if known (e.g., 'Prakash', 'a guy', 'everyone')",
    )
    clothing_items: list[ClothingItem] = Field(
        default_factory=list,
        description="All clothing/accessories worn by this person",
    )
    actions: list[str] = Field(
        default_factory=list,
        description="Actions this person is doing (e.g., 'bowling', 'running', 'gets into accident')",
    )
    action_results: list[str] = Field(
        default_factory=list,
        description="Results of actions (e.g., 'hitting a strike', 'all pins fell')",
    )


class VehicleInQuery(BaseModel):
    """A vehicle mentioned in the query."""

    vehicle_type: str = Field(
        default="", description="Type (e.g., 'car', 'ferrari', 'lamborghini')"
    )
    brand: str = Field(
        default="", description="Brand (e.g., 'Ferrari', 'Lamborghini')"
    )
    model: str = Field(
        default="",
        description="Model if mentioned (e.g., 'Hurricane', 'Model 3')",
    )
    color: str = Field(default="", description="Color (e.g., 'red', 'yellow')")
    actions: list[str] = Field(
        default_factory=list,
        description="What vehicle is doing (e.g., 'passes by', 'crashes')",
    )


class ParsedQuery(BaseModel):
    """Structured representation of extremely complex user search queries.

    Designed to handle paragraph-length queries with:
    - Multiple people with different attributes
    - Body-part specific clothing/accessories
    - Multiple vehicles with brands/colors
    - Temporal descriptions
    - NO hardcoding - all extracted dynamically by LLM

    Example queries this handles:
    - "Prakash wearing blue t-shirt with John Jacobs spectacles and red shoe on left foot,
       green shoe on right foot playing ten-pin bowling hitting a strike where last pin
       takes a moment to fall"
    - "Red Ferrari and yellow Lamborghini Hurricane passing a red signal while a guy
       with Nike blue shoes and VR headset gets into accident"
    """

    # Multiple people with full attributes
    people: list[PersonInQuery] = Field(
        default_factory=list,
        description="All people mentioned with their clothing and actions",
    )

    # All entities extracted from the query (Dynamic)
    entities: list[DynamicEntity] = Field(
        default_factory=list,
        description="ALL entities from query - unlimited types and attributes",
    )

    # Legacy single-person fields (for backwards compatibility)
    person_name: str | None = Field(
        default=None,
        description="Primary person name (use people[] for multiple)",
    )
    clothing_color: str | None = Field(default=None)
    clothing_type: str | None = Field(default=None)
    accessories: list[str] = Field(default_factory=list)

    # Multiple vehicles
    vehicles: list[VehicleInQuery] = Field(
        default_factory=list,
        description="All vehicles mentioned with their brands/colors/actions",
    )

    # Visual elements (expanded by LLM)
    visual_keywords: list[str] = Field(
        default_factory=list,
        description="Expanded visual keywords (e.g., 'South Indian breakfast' -> ['idli', 'dosa'])",
    )

    # Actions (scene-level)
    action_keywords: list[str] = Field(
        default_factory=list, description="All action keywords from query"
    )
    action_result: str | None = Field(
        default=None, description="Primary expected outcome"
    )

    # Objects/Brands
    text_to_find: list[str] = Field(
        default_factory=list,
        description="All brand names and visible text to search",
    )
    objects: list[str] = Field(
        default_factory=list,
        description="Specific objects (e.g., 'bowling pins', 'red signal')",
    )

    # Temporal descriptors
    temporal_hints: list[str] = Field(
        default_factory=list,
        description="Time descriptions (e.g., 'fastly', 'takes a moment', 'at the end')",
    )

    # Location
    location: str | None = Field(
        default=None,
        description="Location (e.g., 'Brunswick sports', 'red signal intersection')",
    )

    # Scene description for very complex queries
    scene_description: str = Field(
        default="", description="LLM-generated summary of what to search for"
    )

    def to_search_text(self) -> str:
        """Aggregates all legacy query components into a single search string.

        Maintains compatibility with older search methods while capturing
        complex multi-person and multi-vehicle semantics.

        Returns:
            A consolidated search string.
        """
        parts = []

        # 1. PEOPLE (with all their attributes)
        for person in self.people:
            if person.name:
                parts.extend(
                    [person.name, person.name]
                )  # Double weight for names
            for item in person.clothing_items:
                if item.brand:
                    parts.append(item.brand)
                if item.color and item.item_type:
                    parts.append(f"{item.color} {item.item_type}")
                elif item.item_type:
                    parts.append(item.item_type)
                if item.details:
                    parts.append(item.details)
                if item.body_part and item.body_part not in [
                    "upper body",
                    "lower body",
                ]:
                    parts.append(
                        f"{item.color} {item.body_part}"
                        if item.color
                        else item.body_part
                    )
            parts.extend(person.actions)
            parts.extend(person.action_results)

        # 2. VEHICLES (brand, model, color, actions)
        for vehicle in self.vehicles:
            if vehicle.brand:
                parts.append(vehicle.brand)
            if vehicle.model:
                parts.append(vehicle.model)
            if vehicle.color:
                parts.append(
                    f"{vehicle.color} {vehicle.vehicle_type or 'vehicle'}"
                )
            elif vehicle.vehicle_type:
                parts.append(vehicle.vehicle_type)
            parts.extend(vehicle.actions)

        # 3. Legacy/fallback fields
        if self.person_name:
            parts.append(self.person_name)
        if self.clothing_color:
            parts.append(self.clothing_color)
        if self.clothing_type:
            parts.append(self.clothing_type)
        parts.extend(self.accessories)

        # 4. Visual keywords
        parts.extend(self.visual_keywords)

        # 5. Actions
        parts.extend(self.action_keywords)
        if self.action_result:
            parts.append(self.action_result)

        # 6. Brands and objects
        parts.extend(self.text_to_find)
        parts.extend(self.objects)

        # 7. Location
        if self.location:
            parts.append(self.location)

        # 8. Scene description (fallback for very complex queries)
        if self.scene_description:
            parts.append(self.scene_description)

        # 9. Temporal hints (for action sequencing)
        parts.extend(self.temporal_hints)

        return " ".join(filter(None, parts))
