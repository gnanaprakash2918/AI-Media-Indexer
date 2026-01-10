
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from core.storage.db import VectorDB
from qdrant_client.http import models

def inspect_data():
    db = VectorDB()
    output = []
    
    output.append("\n--- Inspecting Faces for Cluster 2 ---")
    resp = db.client.scroll(
        collection_name=db.FACES_COLLECTION,
        scroll_filter=models.Filter(
            must=[models.FieldCondition(key="cluster_id", match=models.MatchValue(value=2))]
        ),
        limit=5,
        with_payload=True
    )
    faces = resp[0]
    output.append(f"Faces in Cluster 2: {len(faces)}")
    for face in faces:
        output.append(f"ID: {face.id}, Name: {face.payload.get('name')}, Media: {face.payload.get('media_path')}")

    output.append("\n--- Searching Frames with Cluster 2 ---")
    resp = db.client.scroll(
        collection_name=db.MEDIA_COLLECTION,
        scroll_filter=models.Filter(
            must=[
                models.FieldCondition(
                    key="face_cluster_ids",
                    match=models.MatchAny(any=[2])
                )
            ]
        ),
        limit=5,
        with_payload=True
    )
    frames = resp[0]
    output.append(f"Frames with Cluster 2: {len(frames)}")
    for frame in frames:
        payload = frame.payload or {}
        output.append(f"ID: {frame.id}")
        output.append(f"  Video: {payload.get('video_path')}")
        output.append(f"  Face Clusters: {payload.get('face_cluster_ids')}")
        output.append(f"  Identity Text: {payload.get('identity_text')}")
        output.append("-" * 20)

    output.append("\n--- Searching Frames from Brahmastra Video ---")
    # Try to find frames from the video to see what they have
    resp = db.client.scroll(
        collection_name=db.MEDIA_COLLECTION,
        scroll_filter=models.Filter(
            must=[
                models.FieldCondition(
                    key="video_path",
                    match=models.MatchText(text="BRAHMASTRA") # Loose match to find the video
                )
            ]
        ),
        limit=5,
        with_payload=True
    )
    frames = resp[0]
    output.append(f"Frames from Brahmastra: {len(frames)}")
    if not frames:
         output.append("No frames found with text match 'BRAHMASTRA'. Trying to list all videos...")
         media = db.get_indexed_media()
         for m in media:
             output.append(f"Indexed Media: {m['video_path']}")
    
    for frame in frames:
        payload = frame.payload or {}
        output.append(f"ID: {frame.id}")
        output.append(f"  Video: {payload.get('video_path')}")
        output.append(f"  Face Clusters: {payload.get('face_cluster_ids')}") 
        output.append(f"  Identity Text: {payload.get('identity_text')}")
        output.append("-" * 20)

    with open("debug_output.txt", "w", encoding="utf-8") as f:
        f.write("\n".join(output))
    
    print("Debug output written to debug_output.txt")

if __name__ == "__main__":
    inspect_data()
