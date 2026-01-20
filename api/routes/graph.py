from fastapi import APIRouter, HTTPException, Query, Request
from pydantic import BaseModel
from typing import List, Optional
from core.utils.logger import log

router = APIRouter()

class CoOccurrence(BaseModel):
    name: str
    cluster_id: int
    count: int
    relationship_strength: float

class SocialGraphResponse(BaseModel):
    center_person: str
    center_cluster_id: int
    connections: List[CoOccurrence]

@router.get("/social", response_model=SocialGraphResponse)
async def get_social_graph(
    request: Request,
    name: Optional[str] = None,
    cluster_id: Optional[int] = None,
    min_strength: int = 2
):
    """Get the social graph (co-occurrences) for a specific person.
    
    Args:
        request: FastAPI request object (to access app state).
        name: Name of the person to look up.
        cluster_id: Direct cluster ID lookup (faster).
        min_strength: Minimum number of co-occurrences to be included.
    """
    pipeline = request.app.state.pipeline
    if not pipeline or not pipeline.db:
        raise HTTPException(status_code=503, detail="Database not initialized")

    if not name and cluster_id is None:
        raise HTTPException(status_code=400, detail="Must provide either name or cluster_id")

    # Resolve Name -> Cluster ID if needed
    if cluster_id is None and name:
        cluster_id = pipeline.db.get_cluster_id_by_name(name)
        if cluster_id is None:
             # Try fuzzy lookup
            cluster_id = pipeline.db.fuzzy_get_cluster_id_by_name(name)
    
    if cluster_id is None:
        raise HTTPException(status_code=404, detail=f"Person '{name}' not found")

    # Get the official name for the response
    resolved_name = pipeline.db.get_face_name_by_cluster(cluster_id) or f"Person {cluster_id}"

    try:
        # Fetch co-occurrences from GraphRAG (DB)
        connections_map = pipeline.db.get_person_co_occurrences(cluster_id)
        
        response_connections = []
        for cid, count in connections_map.items():
            if count < min_strength:
                continue
                
            c_name = pipeline.db.get_face_name_by_cluster(cid) or f"Person {cid}"
            
            # Simple normalization for "strength" (0.0 - 1.0) could be added here
            # For now, raw count is the strength
            
            response_connections.append(
                CoOccurrence(
                    name=c_name,
                    cluster_id=cid,
                    count=count,
                    relationship_strength=float(count) 
                )
            )
            
        # Sort by strength
        response_connections.sort(key=lambda x: x.count, reverse=True)

        return SocialGraphResponse(
            center_person=resolved_name,
            center_cluster_id=cluster_id,
            connections=response_connections
        )

    except Exception as e:
        log(f"Social graph lookup failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))
