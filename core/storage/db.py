from qdrant_client import QdrantClient
from qdrant_client.http import models
from sentence_transformers import SentenceTransformer


class VectorDB:
    MEDIA_SEGMENTS_COLLECTION = "media_segments"
    MEDIA_COLLECTION = "media_frames"
    FACES_COLLECTION = "faces"
    DOCS_COLLECTION = "docs"

    # all-MiniLM-L6-v2 output dim
    MEDIA_VECTOR_SIZE = 768
    FACE_VECTOR_SIZE = 128

    # encoder dim
    TEXT_DIM = 384

    def __init__(
        self,
        backend: str = "memory",
        host: str = "localhost",
        port: int = 6333,
        path: str = "qdrant_data",
    ):
        if backend == "memory":
            self.client = QdrantClient(path=path)
            print(f"Initialized embedded Qdrant at path={path}")
        elif backend == "docker":
            try:
                self.client = QdrantClient(host=host, port=port)
                self.client.get_collections()
            except Exception as e:
                print(f"Could not connect to Qdrant: {e}")
                raise ConnectionError("Qdrant connection failed.") from e
            print(f"Connected to Qdrant at {host}:{port}")
        else:
            raise ValueError(f"Unknown backend: {backend!r} (use 'memory' or 'docker')")

        print("Loading SentenceTransformer model...")
        self.encoder = SentenceTransformer("all-MiniLM-L6-v2", device="cpu")

        self._ensure_collections()

    def _ensure_collections(self) -> None:
        if not self.client.collection_exists(self.MEDIA_SEGMENTS_COLLECTION):
            print("media_segments collection not found. Creating it.")
            self.client.create_collection(
                collection_name=self.MEDIA_SEGMENTS_COLLECTION,
                vectors_config=models.VectorParams(
                    size=self.TEXT_DIM,
                    distance=models.Distance.COSINE,
                ),
            )

        if not self.client.collection_exists(self.MEDIA_COLLECTION):
            print("media_frames collection not found. Creating it.")
            self.client.create_collection(
                collection_name=self.MEDIA_COLLECTION,
                vectors_config=models.VectorParams(
                    size=self.MEDIA_VECTOR_SIZE,
                    distance=models.Distance.COSINE,
                ),
            )

        if not self.client.collection_exists(self.FACES_COLLECTION):
            print("faces collection not found. Creating it.")
            self.client.create_collection(
                collection_name=self.FACES_COLLECTION,
                vectors_config=models.VectorParams(
                    size=self.FACE_VECTOR_SIZE,
                    distance=models.Distance.EUCLID,
                ),
            )

        if not self.client.collection_exists(self.DOCS_COLLECTION):
            print("docs collection not found. Creating it.")
            self.client.create_collection(
                collection_name=self.DOCS_COLLECTION,
                vectors_config=models.VectorParams(
                    size=self.TEXT_DIM,
                    distance=models.Distance.COSINE,
                ),
            )

        print("Collections ensured!")

    def list_collections(self):
        return self.client.get_collections()
