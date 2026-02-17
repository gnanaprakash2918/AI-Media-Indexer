"""API routes for council configuration.

Enables frontend to:
- View council models and their states
- Switch between OSS-only, commercial-only, combined modes
- Enable/disable specific models
- Adjust model weights
"""

from __future__ import annotations

from typing import Annotated

from fastapi import APIRouter, Body, HTTPException
from pydantic import BaseModel

from core.processing.council_config import (
    COUNCIL_CONFIG,
    ModelSpec,
    ModelType,
)
from core.utils.logger import get_logger

log = get_logger(__name__)

router = APIRouter(prefix="/councils", tags=["councils"])


class ModelUpdate(BaseModel):
    """Request body for model update."""

    enabled: bool | None = None
    weight: float | None = None


class ModeUpdate(BaseModel):
    """Request body for mode update."""

    mode: str  # 'oss_only', 'commercial_only', 'combined'


class AddModelRequest(BaseModel):
    """Request body for adding a new model."""

    name: str
    model_type: str  # 'oss' or 'commercial'
    model_id: str
    enabled: bool = True
    weight: float = 1.0
    vram_gb: float = 0.0
    description: str = ""


@router.get("")
async def get_councils_config() -> dict:
    """Get full council configuration.

    Returns:
        Complete council configuration including all models.
    """
    return COUNCIL_CONFIG.to_dict()


@router.get("/{council_name}")
async def get_council(council_name: str) -> dict:
    """Get configuration for a specific council.

    Args:
        council_name: 'vlm', 'asr', 'rerank', or 'audio_event'.

    Returns:
        Council configuration with models.
    """
    config = COUNCIL_CONFIG.to_dict()
    if council_name not in config["councils"]:
        raise HTTPException(
            status_code=404,
            detail=f"Council '{council_name}' not found",
        )
    return {
        "mode": config["mode"],
        "council": config["councils"][council_name],
    }


@router.put("/mode")
async def set_council_mode(
    update: Annotated[ModeUpdate, Body()],
) -> dict:
    """Set council operation mode.

    Args:
        update: Mode update request ('oss_only', 'commercial_only', 'combined').

    Returns:
        Updated configuration.
    """
    try:
        COUNCIL_CONFIG.set_mode(update.mode)
        log.info(f"[API] Council mode set to: {update.mode}")
        return {
            "success": True,
            "mode": update.mode,
            "config": COUNCIL_CONFIG.to_dict(),
        }
    except ValueError as e:
        log.warning(f"[API] Invalid council mode '{update.mode}': {e}")
        raise HTTPException(
            status_code=400,
            detail=f"Invalid mode: must be 'oss_only', 'commercial_only', or 'combined'"
        ) from e


@router.patch("/{council_name}/models/{model_name}")
async def update_model(
    council_name: str,
    model_name: str,
    update: Annotated[ModelUpdate, Body()],
) -> dict:
    """Update a model's enabled state or weight.

    Args:
        council_name: Council containing the model.
        model_name: Name of the model to update.
        update: Fields to update (enabled, weight).

    Returns:
        Updated model state.
    """
    if update.enabled is not None:
        success = COUNCIL_CONFIG.set_model_enabled(
            council_name, model_name, update.enabled
        )
        if not success:
            raise HTTPException(
                status_code=404,
                detail=f"Model '{model_name}' not found in '{council_name}'",
            )

    if update.weight is not None:
        success = COUNCIL_CONFIG.set_model_weight(
            council_name, model_name, update.weight
        )
        if not success:
            raise HTTPException(
                status_code=404,
                detail=f"Model '{model_name}' not found in '{council_name}'",
            )

    log.info(f"[API] Updated {council_name}/{model_name}: {update}")
    return {
        "success": True,
        "council": council_name,
        "model": model_name,
        "update": update.model_dump(exclude_none=True),
    }


@router.post("/{council_name}/models")
async def add_model(
    council_name: str,
    request: Annotated[AddModelRequest, Body()],
) -> dict:
    """Add a new model to a council.

    Args:
        council_name: Council to add model to.
        request: Model specification.

    Returns:
        Updated council configuration.
    """
    try:
        model_type = ModelType(request.model_type)
    except ValueError as e:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid model_type: {request.model_type}",
        ) from e

    model = ModelSpec(
        name=request.name,
        model_type=model_type,
        model_id=request.model_id,
        enabled=request.enabled,
        weight=request.weight,
        vram_gb=request.vram_gb,
        description=request.description,
    )

    success = COUNCIL_CONFIG.add_model(council_name, model)
    if not success:
        raise HTTPException(
            status_code=400,
            detail=f"Could not add model to '{council_name}'",
        )

    log.info(f"[API] Added model {request.name} to {council_name}")
    return {
        "success": True,
        "council": council_name,
        "model": request.model_dump(),
    }


@router.get("/{council_name}/enabled")
async def get_enabled_models(council_name: str) -> dict:
    """Get only the enabled models for a council.

    Args:
        council_name: Council name.

    Returns:
        List of enabled models.
    """
    models = COUNCIL_CONFIG.get_enabled(council_name)
    return {
        "council": council_name,
        "enabled_count": len(models),
        "models": [
            {
                "name": m.name,
                "model_id": m.model_id,
                "weight": m.weight,
                "vram_gb": m.vram_gb,
            }
            for m in models
        ],
    }
