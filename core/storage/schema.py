"""Collection schema initialization and migration for Qdrant."""

from __future__ import annotations

from qdrant_client import QdrantClient
from qdrant_client.http import models

from config import settings
from core.storage.constants import (
    AUDIO_EVENTS_COLLECTION,
    FACES_COLLECTION,
    MASKLETS_COLLECTION,
    MEDIA_COLLECTION,
    MEDIA_SEGMENTS_COLLECTION,
    MEDIA_VECTOR_SIZE,
    FACE_VECTOR_SIZE,
    SCENES_COLLECTION,
    SCENELETS_COLLECTION,
    SUMMARIES_COLLECTION,
    TEXT_DIM,
    VIDEO_METADATA_COLLECTION,
    VOICE_COLLECTION,
    VOICE_VECTOR_SIZE,
)
from core.utils.logger import log


def _check_and_fix_collection(
    client: QdrantClient,
    collection_name: str,
    expected_size: int,
    distance: models.Distance = models.Distance.COSINE,
    is_multi_vector: bool = False,
    multi_vector_config: dict | None = None,
) -> None:
    """Check if collection exists and has correct dimension. Recreate if mismatch."""
    try:
        if client.collection_exists(collection_name):
            info = client.get_collection(collection_name)
            if is_multi_vector and multi_vector_config:
                # Multi-vector: check each named vector
                existing = info.config.params.vectors
                if isinstance(existing, dict):
                    for name, cfg in multi_vector_config.items():
                        if name not in existing:
                            log(
                                f"Collection {collection_name} missing vector {name}. Recreating.",
                                level="WARNING",
                            )
                            client.delete_collection(collection_name)
                            break
                        if existing[name].size != cfg.size:
                            log(
                                f"Collection {collection_name} vector {name} dim mismatch: "
                                f"{existing[name].size} != {cfg.size}. Recreating.",
                                level="WARNING",
                            )
                            client.delete_collection(collection_name)
                            break
                    else:
                        return  # All vectors match
                else:
                    log(
                        f"Collection {collection_name} not multi-vector. Recreating.",
                        level="WARNING",
                    )
                    client.delete_collection(collection_name)
            else:
                actual_size = info.config.params.vectors.size
                if actual_size != expected_size:
                    log(
                        f"Collection {collection_name} dim mismatch: "
                        f"{actual_size} != {expected_size}. Recreating.",
                        level="WARNING",
                    )
                    client.delete_collection(collection_name)
                else:
                    return  # Exists and correct
    except Exception as e:
        log(f"Error checking collection {collection_name}: {e}", level="WARNING")

    # Create collection
    if is_multi_vector and multi_vector_config:
        client.create_collection(
            collection_name=collection_name,
            vectors_config=multi_vector_config,
        )
    else:
        client.create_collection(
            collection_name=collection_name,
            vectors_config=models.VectorParams(
                size=expected_size, distance=distance
            ),
        )
    log(f"Created collection: {collection_name}")


def _create_text_index(
    client: QdrantClient,
    collection_name: str,
    field_name: str,
    type: models.TextIndexType = models.TextIndexType.TEXT,
    tokenizer: models.TokenizerType = models.TokenizerType.WORD,
    min_token_len: int = 2,
    max_token_len: int = 20,
    lowercase: bool = True,
) -> None:
    """Create a text payload index with common settings."""
    client.create_payload_index(
        collection_name=collection_name,
        field_name=field_name,
        field_schema=models.TextIndexParams(
            type=type,
            tokenizer=tokenizer,
            min_token_len=min_token_len,
            max_token_len=max_token_len,
            lowercase=lowercase,
        ),
    )


def ensure_all_collections(client: QdrantClient) -> None:
    """Create/verify ALL Qdrant collections and payload indexes."""

    # 1. Media Segments (Text Only)
    _check_and_fix_collection(client, MEDIA_SEGMENTS_COLLECTION, TEXT_DIM)

    # 2. Media Frames (Visual + Metadata)
    _check_and_fix_collection(client, MEDIA_COLLECTION, MEDIA_VECTOR_SIZE)
    client.create_payload_index(
        collection_name=MEDIA_COLLECTION,
        field_name="video_path",
        field_schema=models.PayloadSchemaType.KEYWORD,
    )
    _create_text_index(client, MEDIA_COLLECTION, "transcript")
    client.create_payload_index(
        collection_name=MEDIA_COLLECTION,
        field_name="scan_id",
        field_schema=models.PayloadSchemaType.KEYWORD,
    )
    _create_text_index(
        client, MEDIA_COLLECTION, "ocr_text",
        min_token_len=2, max_token_len=20, lowercase=True,
    )

    # 3. Faces (Euclidean Distance)
    _check_and_fix_collection(
        client, FACES_COLLECTION, FACE_VECTOR_SIZE,
        distance=models.Distance.EUCLID,
    )
    for field, schema in [
        ("video_path", models.PayloadSchemaType.KEYWORD),
        ("cluster_id", models.PayloadSchemaType.INTEGER),
        ("confidence", models.PayloadSchemaType.FLOAT),
    ]:
        client.create_payload_index(
            collection_name=FACES_COLLECTION,
            field_name=field,
            field_schema=schema,
        )

    # 4. Scenelets
    _check_and_fix_collection(
        client, SCENELETS_COLLECTION, TEXT_DIM,
        is_multi_vector=True,
        multi_vector_config={
            "content": models.VectorParams(
                size=TEXT_DIM, distance=models.Distance.COSINE,
            ),
        },
    )
    client.create_payload_index(
        collection_name=SCENELETS_COLLECTION,
        field_name="media_path",
        field_schema=models.PayloadSchemaType.KEYWORD,
    )
    client.create_payload_index(
        collection_name=SCENELETS_COLLECTION,
        field_name="start_time",
        field_schema=models.PayloadSchemaType.FLOAT,
    )

    # 5. Voice Segments
    _check_and_fix_collection(client, VOICE_COLLECTION, VOICE_VECTOR_SIZE)
    for field, schema in [
        ("media_path", models.PayloadSchemaType.KEYWORD),
        ("emotion", models.PayloadSchemaType.KEYWORD),
        ("speaker_label", models.PayloadSchemaType.KEYWORD),
        ("voice_cluster_id", models.PayloadSchemaType.INTEGER),
    ]:
        client.create_payload_index(
            collection_name=VOICE_COLLECTION,
            field_name=field,
            field_schema=schema,
        )

    # 6. Scenes (Multi-Vector)
    visual_features_dim = getattr(settings, "visual_features_dim", 1152)
    video_embedding_dim = getattr(settings, "video_embedding_dim", 1024)
    _check_and_fix_collection(
        client, SCENES_COLLECTION, MEDIA_VECTOR_SIZE,
        is_multi_vector=True,
        multi_vector_config={
            "visual": models.VectorParams(size=TEXT_DIM, distance=models.Distance.COSINE),
            "motion": models.VectorParams(size=TEXT_DIM, distance=models.Distance.COSINE),
            "dialogue": models.VectorParams(size=TEXT_DIM, distance=models.Distance.COSINE),
            "visual_features": models.VectorParams(size=visual_features_dim, distance=models.Distance.COSINE),
            "internvideo": models.VectorParams(size=video_embedding_dim, distance=models.Distance.COSINE),
            "languagebind": models.VectorParams(size=video_embedding_dim, distance=models.Distance.COSINE),
        },
    )

    # 7. Summaries
    _check_and_fix_collection(client, SUMMARIES_COLLECTION, MEDIA_VECTOR_SIZE)

    # 8. Text indexes for hybrid search on frames
    text_fields = [
        "action", "dialogue", "description", "entities",
        "visible_text", "face_names",
        "clothing_colors", "clothing_types", "clothing_descriptions",
        "accessories", "scene_location", "scene_type",
        "object_labels", "dominant_color",
    ]
    for field in text_fields:
        try:
            _create_text_index(
                client, MEDIA_COLLECTION, field,
                min_token_len=2, lowercase=True,
            )
        except Exception:
            pass  # Index may already exist

    # 9. Audio Events (CLAP 512-dim)
    CLAP_DIM = 512
    if not client.collection_exists(AUDIO_EVENTS_COLLECTION):
        client.create_collection(
            collection_name=AUDIO_EVENTS_COLLECTION,
            vectors_config=models.VectorParams(
                size=CLAP_DIM, distance=models.Distance.COSINE,
            ),
        )
        client.create_payload_index(
            collection_name=AUDIO_EVENTS_COLLECTION,
            field_name="media_path",
            field_schema=models.PayloadSchemaType.KEYWORD,
        )
    try:
        for field, schema in [
            ("event_label", models.PayloadSchemaType.KEYWORD),
            ("confidence", models.PayloadSchemaType.FLOAT),
            ("start_time", models.PayloadSchemaType.FLOAT),
        ]:
            client.create_payload_index(
                collection_name=AUDIO_EVENTS_COLLECTION,
                field_name=field,
                field_schema=schema,
            )
    except Exception:
        pass

    # 10. Video Metadata
    if not client.collection_exists(VIDEO_METADATA_COLLECTION):
        client.create_collection(
            collection_name=VIDEO_METADATA_COLLECTION,
            vectors_config=models.VectorParams(
                size=1, distance=models.Distance.COSINE,
            ),
        )
        client.create_payload_index(
            collection_name=VIDEO_METADATA_COLLECTION,
            field_name="media_path",
            field_schema=models.PayloadSchemaType.KEYWORD,
        )

    # 11. Masklets
    _check_and_fix_collection(
        client, MASKLETS_COLLECTION, MEDIA_VECTOR_SIZE,
        distance=models.Distance.COSINE,
    )
    client.create_payload_index(
        collection_name=MASKLETS_COLLECTION,
        field_name="media_path",
        field_schema=models.PayloadSchemaType.KEYWORD,
    )

    log("Qdrant collections and indexes ensured")
