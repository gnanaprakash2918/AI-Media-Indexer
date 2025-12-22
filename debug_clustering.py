import asyncio
import numpy as np
from core.ingestion.pipeline import IngestionPipeline
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics.pairwise import cosine_similarity
from core.utils.logger import logger

async def analyze():
    pipeline = IngestionPipeline()
    faces = pipeline.db.get_all_face_embeddings()
    print(f"Total faces: {len(faces)}")
    
    if not faces:
        return

    # Check dimensions
    dims = len(faces[0]["embedding"])
    print(f"Embedding Dimensions: {dims}")
    
    embeddings = np.array([f["embedding"] for f in faces])
    
    # Check for zero-padding (SFace detection)
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
    embeddings_norm = embeddings / norms

    # Calculate pairwise similarity stats
    sim_matrix = cosine_similarity(embeddings_norm)
    # Remove self-similarity
    np.fill_diagonal(sim_matrix, 0)
    
    print(f"Max Similarity: {np.max(sim_matrix):.4f}")
    print(f"Mean Similarity: {np.mean(sim_matrix):.4f}")
    
    # Test Clustering with various thresholds
    print("\n--- Testing Agglomerative thresholds ---")
    for thresh in [1.0, 1.1, 1.2, 1.3, 1.4, 1.5]:
        clusterer = AgglomerativeClustering(
            n_clusters=None,
            metric="euclidean",
            linkage="average",
            distance_threshold=thresh,
        )
        labels = clusterer.fit_predict(embeddings_norm)
        n_clusters = len(set(labels))
        print(f"Threshold {thresh}: {n_clusters} clusters")

if __name__ == "__main__":
    asyncio.run(analyze())
