
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from core.storage.db import VectorDB
from qdrant_client.http import models

def inspect_data():
    db = VectorDB()
    output = []
    
    stats = db.get_collection_stats()
    output.append(f"Collection Stats: {stats}")
    
    # Get exact path from indexed media
    media = db.get_indexed_media()
    brahmastra_path = None
    for m in media:
        if "BRAHM" in m['video_path']:
             brahmastra_path = m['video_path']
             break
    
    if brahmastra_path:
        output.append(f"Found Brahmastra path: {brahmastra_path}")
        
        # Check frames with EXACT match
        resp = db.client.scroll(
            collection_name=db.MEDIA_COLLECTION,
            scroll_filter=models.Filter(
                must=[
                    models.FieldCondition(
                        key="video_path",
                        match=models.MatchValue(value=brahmastra_path)
                    )
                ]
            ),
            limit=5,
            with_payload=True
        )
        frames = resp[0]
        output.append(f"Frames with Exact Path Match: {len(frames)}")
        for frame in frames:
            payload = frame.payload or {}
            output.append(f"ID: {frame.id}")
            output.append(f"  Face Clusters: {payload.get('face_cluster_ids')}") 
            output.append("-" * 20)
    else:
        output.append("Could not find Brahmastra video in indexed media list.")

    with open("debug_output_2.txt", "w", encoding="utf-8") as f:
        f.write("\n".join(output))
    
    print("Debug output written to debug_output_2.txt")

if __name__ == "__main__":
    inspect_data()
