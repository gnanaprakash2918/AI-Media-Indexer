import sys
import os

# Add project root to path
sys.path.append(os.getcwd())

from core.storage.db import VectorDB
from config import settings

def reset():
    print(f"Connecting to Qdrant ({settings.qdrant_backend})...")
    try:
        db = VectorDB()
        collections = [
            VectorDB.MEDIA_SEGMENTS_COLLECTION,
            VectorDB.MEDIA_COLLECTION,
            VectorDB.FACES_COLLECTION,
            VectorDB.VOICE_COLLECTION
        ]
        
        for name in collections:
            print(f"Deleting collection: {name}")
            try:
                db.client.delete_collection(name)
            except Exception as e:
                print(f"Error deleting {name}: {e}")
                
        print("Reset complete.")
    except Exception as e:
        print(f"Connection failed: {e}")

if __name__ == "__main__":
    reset()
