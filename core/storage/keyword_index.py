"""Persistent BM25 index for hybrid search.

Enables keyword search for proper names and terms that semantic search misses.
Persists to disk alongside Qdrant for durability.
"""

import pickle
from pathlib import Path
from typing import Any

from rank_bm25 import BM25Okapi

from core.utils.logger import get_logger

log = get_logger(__name__)


class KeywordIndex:
    """Persistent BM25 index for hybrid search.

    Usage:
        index = KeywordIndex(Path("data/bm25.pkl"))
        index.load()
        index.add_document("doc1", "Prakash bowling strike")
        results = index.search("Prakash", top_k=50)
    """

    def __init__(self, index_path: Path):
        """Initialize keyword index.

        Args:
            index_path: Path to persist the BM25 index.
        """
        self.index_path = Path(index_path)
        self.bm25: BM25Okapi | None = None
        self.corpus: list[str] = []
        self.doc_ids: list[str] = []
        self._dirty = False

    def load(self) -> bool:
        """Load index from disk if exists.

        Returns:
            True if index was loaded successfully.
        """
        if not self.index_path.exists():
            log.info("[BM25] No existing index found, starting fresh")
            return False

        try:
            with open(self.index_path, "rb") as f:
                data = pickle.load(f)
            self.bm25 = data["bm25"]
            self.corpus = data["corpus"]
            self.doc_ids = data["doc_ids"]
            log.info(f"[BM25] Loaded index with {len(self.corpus)} documents")
            return True
        except Exception as e:
            log.error(f"[BM25] Failed to load index: {e}")
            return False

    def save(self) -> bool:
        """Persist index to disk.

        Returns:
            True if saved successfully.
        """
        if self.bm25 is None:
            return False

        try:
            self.index_path.parent.mkdir(parents=True, exist_ok=True)
            with open(self.index_path, "wb") as f:
                pickle.dump(
                    {
                        "bm25": self.bm25,
                        "corpus": self.corpus,
                        "doc_ids": self.doc_ids,
                    },
                    f,
                )
            self._dirty = False
            log.info(f"[BM25] Saved index with {len(self.corpus)} documents")
            return True
        except Exception as e:
            log.error(f"[BM25] Failed to save index: {e}")
            return False

    def build_from_documents(self, documents: list[dict]) -> int:
        """Build index from a list of documents.

        Args:
            documents: List of dicts with 'id' and 'text' keys.

        Returns:
            Number of documents indexed.
        """
        self.corpus = []
        self.doc_ids = []

        for doc in documents:
            doc_id = doc.get("id", "")
            text = doc.get("text", "")
            if doc_id and text:
                self.doc_ids.append(str(doc_id))
                self.corpus.append(text)

        if self.corpus:
            tokenized = [doc.lower().split() for doc in self.corpus]
            self.bm25 = BM25Okapi(tokenized)
            self._dirty = True
            self.save()

        log.info(f"[BM25] Built index from {len(self.corpus)} documents")
        return len(self.corpus)

    def add_document(self, doc_id: str, text: str) -> None:
        """Add a single document to the index.

        Note: This rebuilds the BM25 index (necessary for BM25Okapi).

        Args:
            doc_id: Unique document identifier.
            text: Document text content.
        """
        if not text.strip():
            return

        self.doc_ids.append(str(doc_id))
        self.corpus.append(text)

        # Rebuild BM25 (necessary for BM25Okapi)
        tokenized = [doc.lower().split() for doc in self.corpus]
        self.bm25 = BM25Okapi(tokenized)
        self._dirty = True

    def search(
        self,
        query: str,
        top_k: int = 50,
    ) -> list[tuple[str, float]]:
        """Search and return (doc_id, score) tuples.

        Args:
            query: Search query string.
            top_k: Maximum results to return.

        Returns:
            List of (doc_id, score) tuples, sorted by score descending.
        """
        if self.bm25 is None or not self.corpus:
            return []

        tokenized_query = query.lower().split()
        scores = self.bm25.get_scores(tokenized_query)

        # Get top-k indices
        top_indices = scores.argsort()[::-1][:top_k]

        results = []
        for idx in top_indices:
            score = float(scores[idx])
            if score > 0:
                results.append((self.doc_ids[idx], score))

        return results

    def remove_document(self, doc_id: str) -> bool:
        """Remove a document from the index.

        Args:
            doc_id: Document ID to remove.

        Returns:
            True if document was found and removed.
        """
        try:
            idx = self.doc_ids.index(str(doc_id))
            self.doc_ids.pop(idx)
            self.corpus.pop(idx)

            # Rebuild BM25
            if self.corpus:
                tokenized = [doc.lower().split() for doc in self.corpus]
                self.bm25 = BM25Okapi(tokenized)
            else:
                self.bm25 = None

            self._dirty = True
            return True
        except ValueError:
            return False

    def clear(self) -> None:
        """Clear the entire index."""
        self.bm25 = None
        self.corpus = []
        self.doc_ids = []
        self._dirty = True

    def __len__(self) -> int:
        """Return number of documents in index."""
        return len(self.corpus)

    def __contains__(self, doc_id: str) -> bool:
        """Check if document is in index."""
        return str(doc_id) in self.doc_ids

    def get_stats(self) -> dict[str, Any]:
        """Get index statistics."""
        return {
            "document_count": len(self.corpus),
            "index_path": str(self.index_path),
            "is_loaded": self.bm25 is not None,
            "needs_save": self._dirty,
        }
