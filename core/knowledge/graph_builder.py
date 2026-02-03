"""Graph Builder: Construct the Time-Weighted Knowledge Graph.

Ingests analysis results and builds nodes/edges in Neo4j.
"""

import uuid
from typing import Optional

from core.knowledge.graph_store import get_graph_store
from core.schemas import FrameAnalysis, MediaFile, SceneContext
from core.utils.logger import get_logger

log = get_logger(__name__)


class GraphBuilder:
    """Builds the Knowledge Graph from analysis data."""

    def __init__(self):
        self.store = get_graph_store()

    def process_video_node(self, media_file: MediaFile) -> str:
        """Create or update a Video node. Returns video_id."""
        if not self.store:
            return ""

        query = """
        MERGE (v:Video {path: $path})
        SET v.filename = $filename,
            v.duration = $duration,
            v.created_at = $created_at
        RETURN v.path as id
        """
        params = {
            "path": media_file.path,
            "filename": media_file.filename,
            "duration": media_file.metadata.duration or 0,
            "created_at": str(media_file.metadata.created_at or "")
        }
        self.store.query(query, params)
        return media_file.path

    def process_scene(
        self,
        video_path: str,
        scene_id: str,
        start_time: float,
        end_time: float,
        analysis: FrameAnalysis,
        prev_scene_id: Optional[str] = None
    ) -> str:
        """Ingest a single scene into the graph."""
        if not self.store:
            return ""

        # 1. Create Scene Node
        scene_query = """
        MATCH (v:Video {path: $video_path})
        MERGE (s:Scene {id: $scene_id})
        SET s.start_time = $start,
            s.end_time = $end,
            s.description = $desc,
            s.location = $location
        MERGE (v)-[:CONTAINS]->(s)
        """

        # Safe description extraction
        desc = analysis.to_search_content()
        location = analysis.scene.location if isinstance(analysis.scene, SceneContext) else ""

        self.store.query(scene_query, {
            "video_path": video_path,
            "scene_id": scene_id,
            "start": start_time,
            "end": end_time,
            "desc": desc,
            "location": location
        })

        # 2. Link to Previous Scene (Temporal Chain)
        if prev_scene_id:
            link_query = """
            MATCH (prev:Scene {id: $prev_id})
            MATCH (curr:Scene {id: $curr_id})
            MERGE (prev)-[r:NEXT_SCENE]->(curr)
            SET r.time_gap = curr.start_time - prev.end_time
            """
            self.store.query(link_query, {"prev_id": prev_scene_id, "curr_id": scene_id})

        # 3. Extract & Link Entities (Production Grade)
        self._extract_entities(scene_id, analysis, start_time, end_time)

        return scene_id

    def _extract_entities(self, scene_id: str, analysis: FrameAnalysis, start: float, end: float):
        """Extract People, Objects, Actions and link to Scene with weighted edges."""
        # A. Identities (Person) - Using Face Clusters (Primary) & Names (Secondary)
        if analysis.face_cluster_ids:
            for cluster_id in analysis.face_cluster_ids:
                person_id = f"person_cluster_{cluster_id}"
                # Get name if available
                # In prod, we'd query the DB for the name, or pass it in.
                query = """
                MATCH (s:Scene {id: $scene_id})
                MERGE (p:Person {id: $person_id})
                SET p.cluster_id = $cluster_id
                MERGE (p)-[r:APPEARED_IN]->(s)
                SET r.start = $start, r.end = $end
                """
                self.store.query(query, {
                    "scene_id": scene_id,
                    "person_id": person_id,
                    "cluster_id": cluster_id,
                    "start": start,
                    "end": end
                })

        # B. Rich Actions (Semantic Nodes)
        # "Walking" -> (Action:Walking)
        # We normalize to lowercase to create shared nodes.
        if analysis.action:
            # Simple NLP normalization (first 3 words or full string if short)
            action_raw = analysis._to_str(analysis.action).lower().strip()
            # Heuristic: split by comma, take first part
            action_short = action_raw.split(',')[0].strip()

            action_id = f"action_{uuid.uuid5(uuid.NAMESPACE_DNS, action_short)}"
            query = """
            MATCH (s:Scene {id: $scene_id})
            MERGE (a:Action {id: $action_id})
            SET a.description = $desc
            MERGE (s)-[:DEPICTS]->(a)
            """
            self.store.query(query, {"scene_id": scene_id, "action_id": action_id, "desc": action_short})

        # C. Detailed Objects (With confidence if available)
        for entity in analysis.entities:
            name = ""
            details = ""
            if isinstance(entity, str):
                name = entity
            elif isinstance(entity, dict):
                name = entity.get("name", "")
                details = entity.get("visual_details", "")
            elif hasattr(entity, "name"):
                name = entity.name
                details = getattr(entity, "visual_details", "")

            if name:
                # Normalize object name
                name_clean = name.lower().strip()
                obj_id = f"obj_{name_clean.replace(' ', '_')}"

                # (Object)-[PRESENT_IN {details}]->(Scene)
                query = """
                MATCH (s:Scene {id: $scene_id})
                MERGE (o:Object {id: $obj_id})
                SET o.name = $name
                MERGE (o)-[r:PRESENT_IN]->(s)
                SET r.details = $details
                """
                self.store.query(query, {"scene_id": scene_id, "obj_id": obj_id, "name": name_clean, "details": details})

        # D. Audio/Mood Context (Abusing Audio Analysis)
        # If analysis has scene context with mood/audio info
        if analysis.scene:
            scene_ctx = analysis.scene
            if isinstance(scene_ctx, dict):
                mood = scene_ctx.get("mood")
            elif hasattr(scene_ctx, "mood"):
                mood = scene_ctx.mood
            else:
                mood = None

            if mood:
                mood_clean = mood.lower().strip()
                mood_id = f"mood_{mood_clean}"
                query = """
                MATCH (s:Scene {id: $scene_id})
                MERGE (m:Mood {id: $mood_id})
                SET m.name = $mood
                MERGE (s)-[:HAS_MOOD]->(m)
                """
                self.store.query(query, {"scene_id": scene_id, "mood_id": mood_id, "mood": mood_clean})

        # E. OCR Text (Abusing explicit text signals)
        # If analysis has visible text, creates Text nodes
        # We access visible_text from the scene context inside analysis
        scene_ctx = analysis.scene
        if isinstance(scene_ctx, SceneContext) and scene_ctx.visible_text:
             text_content = scene_ctx.visible_text
             # Heuristic: Only reasonable length text
             for txt_item in text_content: # visible_text is list[str]
                if len(txt_item) > 2:
                     text_id = f"text_{uuid.uuid5(uuid.NAMESPACE_DNS, txt_item[:50])}"
                     query = """
                     MATCH (s:Scene {id: $scene_id})
                     MERGE (t:Text {content: $content})
                     MERGE (s)-[:DISPLAYS_TEXT]->(t)
                     """
                     self.store.query(query, {"scene_id": scene_id, "content": txt_item})

    def process_masklets(self, video_path: str, masklets: list[dict]):
        """Ingest SAM 3 Masklets (Precision Object Tracks) into Graph."""
        if not self.store or not masklets:
            return

        for m in masklets:
            # m = {id, start_time, end_time, concept, confidence, ...}
            concept = m.get("concept", "object").lower().strip()
            masklet_id = f"masklet_{m.get('id')}"

            # Create Tracked Object Node
            # We treat Masklets as high-confidence "PrecisionObject" nodes
            query = """
            MATCH (v:Video {path: $video_path})
            MERGE (o:PrecisionObject {id: $masklet_id})
            SET o.name = $concept,
                o.confidence = $conf,
                o.source = 'sam3'
            
            # Link to Video
            MERGE (o)-[:TRACKED_IN]->(v)
            
            # Link to overlapping Scenes (Temporal Projection)
            WITH o, v, $start as m_start, $end as m_end
            MATCH (v)-[:CONTAINS]->(s:Scene)
            WHERE s.start_time <= m_end AND s.end_time >= m_start
            MERGE (o)-[:PRESENT_IN_SCENE]->(s)
            """

            self.store.query(query, {
                "video_path": video_path,
                "masklet_id": masklet_id,
                "concept": concept,
                "conf": m.get("confidence", 1.0),
                "start": m.get("start_time"),
                "end": m.get("end_time")
            })
