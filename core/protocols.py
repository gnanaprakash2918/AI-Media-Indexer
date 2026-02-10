"""Core protocols for SOLID compliance.

Defines runtime-checkable Protocol classes enabling:
  - OCP: New embedding models only need to implement EmbeddingProvider.
  - LSP: Any class satisfying a protocol can substitute transparently.
  - ISP: Callers depend on narrow interfaces, not 139-method VectorDB.
"""

from __future__ import annotations

from typing import Any, Protocol, runtime_checkable


@runtime_checkable
class EmbeddingProvider(Protocol):
    """OCP: Any embedding backend implements this narrow interface.

    Allows swapping SentenceTransformer → OpenAI → Cohere etc. without
    modifying search or ingestion code.
    """

    def encode_texts(
        self,
        texts: str | list[str],
        *,
        batch_size: int = 1,
        is_query: bool = False,
        job_id: str | None = None,
    ) -> list[list[float]]:
        """Encode one or more texts into embedding vectors."""
        ...

    def get_embedding(self, text: str) -> list[float]:
        """Convenience: encode a single query string."""
        ...


@runtime_checkable
class SearchProvider(Protocol):
    """LSP: Interchangeable search backends.

    Both SearchAgent (agentic) and any future simple/hybrid searcher
    satisfy this contract.
    """

    async def search(
        self,
        query: str,
        limit: int = 20,
        **kwargs: Any,
    ) -> dict[str, Any]:
        """Execute a search and return a result dict."""
        ...


@runtime_checkable
class MediaProcessor(Protocol):
    """ISP: Narrow interface for pipeline stage processors.

    Each pipeline stage (audio, voice, frames, scenes) implements
    this contract so the orchestrator depends on the interface,
    not the concrete class.
    """

    async def process(self, path: Any, *, job_id: str | None = None) -> None:
        """Process a media file or path."""
        ...
