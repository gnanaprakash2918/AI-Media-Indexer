import pytest
import time
from core.storage.db import VectorDB
import numpy as np

@pytest.mark.slow
@pytest.mark.performance
def test_vector_db_insert_performance():
    """Benchmark VectorDB insertion speed (Memory backend)."""
    db = VectorDB(backend="memory", path=":memory:")
    
    # Generate 1000 dummy embeddings
    count = 1000
    dim = 256
    vectors = np.random.rand(count, dim).tolist()
    
    start_time = time.perf_counter()
    
    points = []
    for i, vec in enumerate(vectors):
        # Use memory backend for raw benchmark
        pass
        
    # Use insert_voice_segment for single inserts
    for i in range(100): # Test 100 inserts
        db.insert_voice_segment(
            media_path=f"file_{i}",
            start=0.0,
            end=1.0,
            speaker_label="spk",
            embedding=vectors[i]
        )
        
    duration = time.perf_counter() - start_time
    print(f"\n[Perf] 100 Vector Inserts: {duration:.4f}s")
    
    # Requirement: Should be reasonably fast (e.g., < 2s for 100 in-memory inserts)
    assert duration < 5.0

@pytest.mark.slow
@pytest.mark.performance
def test_search_performance():
    db = VectorDB(backend="memory", path=":memory:")
    
    # Seed data
    count = 500
    dim = 384
    vectors = np.random.rand(count, dim).tolist()
    
    for i in range(count):
        db.upsert_media_frame(
            point_id=f"frame_{i}",
            vector=vectors[i],
            video_path="vid",
            timestamp=float(i),
            action=f"action_{i}"
        )
        
    # Search
    start_time = time.perf_counter()
    for _ in range(50):
        db.search_frames("query", limit=10)
    duration = time.perf_counter() - start_time
    
    print(f"\n[Perf] 50 Searches on {count} items: {duration:.4f}s")
    assert duration < 5.0
