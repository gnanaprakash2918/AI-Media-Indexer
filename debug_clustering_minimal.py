import sys
import numpy as np
import warnings
from qdrant_client import QdrantClient

# Suppress warnings
warnings.filterwarnings("ignore")

def analyze():
    print("Connecting to Qdrant...")
    # Hardcoded defaults from config
    client = QdrantClient(host="localhost", port=6333)
    
    print("Fetching faces...")
    # Scroll all points
    faces = []
    offset = None
    while True:
        resp = client.scroll(
            collection_name="faces",
            limit=100,
            with_payload=True,
            with_vectors=True,
            scroll_filter=None,
            offset=offset
        )
        points, next_offset = resp
        faces.extend(points)
        offset = next_offset
        if offset is None:
            break
            
    print(f"Total faces: {len(faces)}")
    if not faces:
        return

    # Check dimensions
    dims = len(faces[0].vector)
    print(f"Embedding Dimensions: {dims}")
    
    embeddings = np.array([f.vector for f in faces])
    
    # Check for zero-padding
    if dims == 512:
        tail_energy = np.sum(np.abs(embeddings[:, 128:]))
        print(f"Tail Energy (dims 128-512): {tail_energy:.4f}")
        if tail_energy < 1e-3:
            print("WARNING: DETECTED SFACE (128d) PADDED TO 512d! Quality will be low.")
            embeddings = embeddings[:, :128]
        else:
            print("Confirmed InsightFace (512d).")

    # Normalize
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    norms[norms == 0] = 1
    embeddings_norm = embeddings / norms

    # Calculate pairwise similarity stats
    # Manual cosine sim implementation to avoid sklearn import if missing
    sim_matrix = np.dot(embeddings_norm, embeddings_norm.T)
    np.fill_diagonal(sim_matrix, 0)
    
    max_sim = np.max(sim_matrix)
    mean_sim = np.mean(sim_matrix)
    print(f"Max Similarity: {max_sim:.4f}")
    print(f"Mean Similarity: {mean_sim:.4f}")
    
    # Simple histogram of similarities
    print("\nSimilarity Distribution (Sample):")
    sims = sim_matrix[sim_matrix > 0]
    if len(sims) > 0:
        for t in [0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3]:
             count = np.sum(sims > t)
             print(f"Pairs > {t:.1f}: {count}")

    # Test Agglomerative Clustering
    try:
        from sklearn.cluster import AgglomerativeClustering
        print("\n--- Testing Agglomerative thresholds ---")
        for thresh in [1.0, 1.1, 1.2, 1.3, 1.4]:
            clusterer = AgglomerativeClustering(
                n_clusters=None,
                metric="euclidean",
                linkage="average",
                distance_threshold=thresh,
            )
            labels = clusterer.fit_predict(embeddings_norm)
            n_clusters = len(set(labels))
            print(f"Threshold {thresh}: {n_clusters} clusters")
    except ImportError:
        print("sklearn not installed, skipping clustering simulation.")

if __name__ == "__main__":
    analyze()
