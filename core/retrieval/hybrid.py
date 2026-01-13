"""Hybrid search combining vector and keyword search with RRF fusion.

Combines semantic vector search (Qdrant) with keyword search (BM25)
using Reciprocal Rank Fusion for improved retrieval accuracy.
"""

from collections import defaultdict
from pathlib import Path
from typing import Any

from core.storage.keyword_index import KeywordIndex
from core.utils.logger import get_logger

log = get_logger(__name__)


class HybridSearcher:
    """Combines vector search + BM25 keyword search with RRF fusion.

    Usage:
        searcher = HybridSearcher(db, Path("data/bm25.pkl"))
        results = await searcher.search(
            query="Prakash bowling",
            vector_weight=0.6,
            keyword_weight=0.4
        )
    """

    def __init__(self, db: Any, bm25_path: Path | None = None):
        """Initialize hybrid searcher.

        Args:
            db: VectorDB instance for semantic search.
            bm25_path: Path for BM25 index persistence.
        """
        self.db = db
        if bm25_path is None:
            bm25_path = Path("data/bm25_index.pkl")
        self.keyword_index = KeywordIndex(bm25_path)
        self.keyword_index.load()

    async def search(
        self,
        query: str,
        limit: int = 50,
        vector_weight: float = 0.6,
        keyword_weight: float = 0.4,
        video_id: str | None = None,
    ) -> list[dict]:
        """Perform hybrid search with RRF fusion.

        Args:
            query: Search query string.
            limit: Maximum results to return.
            vector_weight: Weight for vector search (0-1).
            keyword_weight: Weight for BM25 search (0-1).
            video_id: Optional filter by video.

        Returns:
            List of results with fused scores and source info.
        """
        # Normalize weights
        total_weight = vector_weight + keyword_weight
        if total_weight > 0:
            vector_weight = vector_weight / total_weight
            keyword_weight = keyword_weight / total_weight

        # Vector search
        vector_results = await self._vector_search(query, limit * 2, video_id)

        # Keyword search
        keyword_results = self.keyword_index.search(query, top_k=limit * 2)

        # RRF Fusion
        fused = self._rrf_fusion(
            vector_results=vector_results,
            keyword_results=keyword_results,
            vector_weight=vector_weight,
            keyword_weight=keyword_weight,
        )

        log.debug(
            f"[Hybrid] query='{query[:50]}' "
            f"vector={len(vector_results)} keyword={len(keyword_results)} "
            f"fused={len(fused)}"
        )

        return fused[:limit]

    async def _vector_search(
        self,
        query: str,
        limit: int,
        video_id: str | None = None,
    ) -> list[dict]:
        """Perform vector search via Qdrant.

        Args:
            query: Search query.
            limit: Max results.
            video_id: Optional video filter.

        Returns:
            List of dicts with 'id' and 'score'.
        """
        try:
            # Get query embedding
            query_embedding = await self.db.get_embedding(query)

            # Search scenes
            results = self.db.search_scenes(
                query_embedding,
                limit=limit,
                video_id=video_id,
            )

            return [
                {
                    "id": r.id if hasattr(r, "id") else str(r),
                    "score": r.score if hasattr(r, "score") else 0.0,
                    "payload": r.payload if hasattr(r, "payload") else {},
                }
                for r in results
            ]
        except Exception as e:
            log.error(f"[Hybrid] Vector search failed: {e}")
            return []

    def _rrf_fusion(
        self,
        vector_results: list[dict],
        keyword_results: list[tuple[str, float]],
        vector_weight: float = 0.6,
        keyword_weight: float = 0.4,
        k: int = 60,
    ) -> list[dict]:
        """Reciprocal Rank Fusion of two result sets.

        RRF Score = sum(weight / (k + rank))

        Args:
            vector_results: Results from vector search.
            keyword_results: Results from BM25 (doc_id, score tuples).
            vector_weight: Weight for vector scores.
            keyword_weight: Weight for keyword scores.
            k: RRF constant (default 60).

        Returns:
            Fused and sorted results.
        """
        scores: dict[str, dict] = defaultdict(
            lambda: {
                "rrf_score": 0.0,
                "vector_rank": None,
                "vector_score": None,
                "keyword_rank": None,
                "keyword_score": None,
                "payload": {},
            }
        )

        # Vector scores
        for rank, result in enumerate(vector_results):
            doc_id = str(result.get("id", ""))
            if not doc_id:
                continue
            scores[doc_id]["rrf_score"] += vector_weight / (k + rank + 1)
            scores[doc_id]["vector_rank"] = rank + 1
            scores[doc_id]["vector_score"] = result.get("score", 0.0)
            scores[doc_id]["payload"] = result.get("payload", {})

        # Keyword scores
        for rank, (doc_id, kw_score) in enumerate(keyword_results):
            scores[doc_id]["rrf_score"] += keyword_weight / (k + rank + 1)
            scores[doc_id]["keyword_rank"] = rank + 1
            scores[doc_id]["keyword_score"] = kw_score

        # Sort by fused score
        fused = [
            {"id": doc_id, **data}
            for doc_id, data in scores.items()
        ]
        fused.sort(key=lambda x: x["rrf_score"], reverse=True)

        return fused

    def index_document(self, doc_id: str, text: str) -> None:
        """Add a document to the keyword index.

        Args:
            doc_id: Document ID.
            text: Text content to index.
        """
        self.keyword_index.add_document(doc_id, text)

    def build_index_from_db(self) -> int:
        """Build BM25 index from all documents in VectorDB.

        Returns:
            Number of documents indexed.
        """
        try:
            # Get all transcripts and descriptions from DB
            documents = []

            # Get from media_frames collection
            frames = self.db.get_all_frame_texts()
            for frame in frames:
                doc_id = frame.get("id", "")
                text_parts = []
                if frame.get("transcript"):
                    text_parts.append(frame["transcript"])
                if frame.get("description"):
                    text_parts.append(frame["description"])
                if frame.get("ocr_text"):
                    text_parts.append(frame["ocr_text"])
                if text_parts:
                    documents.append({
                        "id": doc_id,
                        "text": " ".join(text_parts),
                    })

            count = self.keyword_index.build_from_documents(documents)
            log.info(f"[Hybrid] Built BM25 index with {count} documents")
            return count

        except Exception as e:
            log.error(f"[Hybrid] Failed to build index from DB: {e}")
            return 0

    def save_index(self) -> bool:
        """Save the keyword index to disk."""
        return self.keyword_index.save()

    def get_stats(self) -> dict:
        """Get hybrid search statistics."""
        return {
            "keyword_index": self.keyword_index.get_stats(),
            "vector_db_connected": self.db is not None,
        }
