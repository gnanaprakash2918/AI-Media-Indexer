"""External Knowledge Enrichment via Brave Search.

Provides external context for unknown entities (faces, locations, topics)
by querying the web. Gracefully degrades when API keys are not configured.
"""

from __future__ import annotations

import asyncio
from typing import Any

from pydantic import BaseModel, Field

from config import settings
from core.utils.logger import log


class BraveSearchResult(BaseModel):
    """Single result from Brave Search."""
    title: str
    url: str
    description: str
    age: str | None = None


class BraveSearchClient:
    """Async client for Brave Search API.
    
    Provides web search for enriching unknown entities with external knowledge.
    Gracefully handles missing API keys by returning empty results.
    
    Usage:
        client = BraveSearchClient()
        if client.is_available:
            results = await client.search("Who is Sundar Pichai?")
    """
    
    def __init__(self, api_key: str | None = None):
        """Initialize Brave Search client.
        
        Args:
            api_key: Brave Search API key. Falls back to settings.brave_api_key.
        """
        self.api_key = api_key or getattr(settings, 'brave_api_key', None)
        self._client = None
        
    @property
    def is_available(self) -> bool:
        """Check if Brave Search is configured and available."""
        return bool(self.api_key)
    
    async def search(
        self,
        query: str,
        count: int = 5,
        search_type: str = "web",
    ) -> list[BraveSearchResult]:
        """Perform a web search using Brave Search API.
        
        Args:
            query: Search query string.
            count: Number of results to return (max 20).
            search_type: Type of search (web, news, images).
            
        Returns:
            List of search results. Empty if API unavailable.
        """
        if not self.is_available:
            log("Brave Search not configured (no API key)", level="DEBUG")
            return []
        
        try:
            # Lazy import to avoid dependency issues
            from brave_search import BraveSearch
            
            client = BraveSearch(api_key=self.api_key)
            
            # Run sync client in thread pool
            loop = asyncio.get_event_loop()
            response = await loop.run_in_executor(
                None,
                lambda: client.search(query, count=min(count, 20))
            )
            
            results = []
            web_results = response.get("web", {}).get("results", [])
            
            for item in web_results[:count]:
                results.append(BraveSearchResult(
                    title=item.get("title", ""),
                    url=item.get("url", ""),
                    description=item.get("description", ""),
                    age=item.get("age"),
                ))
            
            return results
            
        except ImportError:
            log("brave-search package not installed. Run: pip install brave-search", level="WARNING")
            return []
        except Exception as e:
            log(f"Brave Search error: {e}", level="ERROR")
            return []


class EntityEnricher:
    """Enriches unknown entities with external knowledge.
    
    Uses Brave Search (or other providers) to:
    - Identify unknown faces by context
    - Get information about locations
    - Expand topic context for better search
    """
    
    def __init__(self):
        self.brave = BraveSearchClient()
        self._cache: dict[str, Any] = {}
    
    @property
    def is_available(self) -> bool:
        """Check if external enrichment is available."""
        # Check if external search is enabled in settings
        enabled = getattr(settings, 'enable_external_search', False)
        return enabled and self.brave.is_available
    
    async def enrich_unknown_face(
        self,
        context: str,
        image_description: str | None = None,
    ) -> dict[str, Any]:
        """Attempt to identify an unknown face using context.
        
        Args:
            context: Contextual information (video title, scene description).
            image_description: Visual description of the person.
            
        Returns:
            Dict with possible_matches and confidence.
        """
        if not self.is_available:
            return {"possible_matches": [], "confidence": 0.0, "source": "unavailable"}
        
        # Build search query from context
        query_parts = []
        if context:
            query_parts.append(context)
        if image_description:
            query_parts.append(image_description)
        
        search_query = f"who is {' '.join(query_parts)}"
        
        # Check cache
        cache_key = search_query.lower().strip()
        if cache_key in self._cache:
            return self._cache[cache_key]
        
        # Search
        results = await self.brave.search(search_query, count=5)
        
        if not results:
            return {"possible_matches": [], "confidence": 0.0}
        
        # Extract potential names from titles and descriptions
        possible_matches = []
        for result in results:
            # Simple heuristic: look for proper nouns in title
            words = result.title.split()
            for i, word in enumerate(words):
                if word[0].isupper() and len(word) > 1:
                    # Check if next word is also capitalized (full name)
                    if i + 1 < len(words) and words[i + 1][0].isupper():
                        possible_matches.append(f"{word} {words[i + 1]}")
                    else:
                        possible_matches.append(word)
        
        # Deduplicate and rank by frequency
        from collections import Counter
        name_counts = Counter(possible_matches)
        top_matches = [name for name, _ in name_counts.most_common(3)]
        
        result = {
            "possible_matches": top_matches,
            "confidence": 0.6 if top_matches else 0.0,
            "source": "brave_search",
            "raw_results": [r.model_dump() for r in results[:3]],
        }
        
        # Cache result
        self._cache[cache_key] = result
        
        return result
    
    async def enrich_location(
        self,
        location_hint: str,
    ) -> dict[str, Any]:
        """Get additional context about a location.
        
        Args:
            location_hint: Location name or description.
            
        Returns:
            Dict with location info.
        """
        if not self.is_available:
            return {"info": None, "confidence": 0.0}
        
        search_query = f"{location_hint} location landmark"
        results = await self.brave.search(search_query, count=3)
        
        if not results:
            return {"info": None, "confidence": 0.0}
        
        # Combine descriptions
        info = " ".join([r.description for r in results[:2]])
        
        return {
            "info": info,
            "confidence": 0.5,
            "source": "brave_search",
        }
    
    async def enrich_topic(
        self,
        topic: str,
    ) -> dict[str, Any]:
        """Get external context about a topic for better search.
        
        Args:
            topic: Topic or subject to research.
            
        Returns:
            Dict with topic context.
        """
        if not self.is_available:
            return {"context": "", "related_terms": []}
        
        results = await self.brave.search(topic, count=5)
        
        if not results:
            return {"context": "", "related_terms": []}
        
        # Build context from results
        context_parts = [r.description for r in results[:3] if r.description]
        context = " ".join(context_parts)[:500]  # Limit length
        
        # Extract related terms (simple word extraction)
        all_text = " ".join([r.title + " " + r.description for r in results])
        words = all_text.split()
        
        # Filter to notable terms (capitalized, longer words)
        related = set()
        for word in words:
            clean = word.strip(".,!?()[]")
            if len(clean) > 4 and clean[0].isupper():
                related.add(clean)
        
        return {
            "context": context,
            "related_terms": list(related)[:10],
            "source": "brave_search",
        }


# Global instance
enricher = EntityEnricher()
