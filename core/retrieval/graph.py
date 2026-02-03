"""Graph Searcher: Execute retrieval against the Neo4j Knowledge Graph.

Converts intent into Cypher queries to find scenes matching temporal or semantic patterns.
"""

from typing import Any, Dict, List

from core.knowledge.graph_store import get_graph_store
from core.utils.logger import get_logger
from llm.factory import LLMFactory

log = get_logger(__name__)

class GraphSearcher:
    """Exploits the Graph for complex retrieval."""

    def __init__(self):
        self.store = get_graph_store()
        # We can use a specialized small LLM for Cypher generation if needed
        self.llm = LLMFactory.create_llm()

    async def search(self, query: str, entities: List[str] = None, actions: List[str] = None) -> List[Dict[str, Any]]:
        """Main entry point for Graph Search.
        For Phase 1, we implement a robust heuristic search:
        - Find connection between entities.
        - Find scenes with specific actions/objects.
        """
        if not self.store:
            return []

        results = []

        # Strategy 1: "Abusing" Entity Co-occurrence (Who appeared with Who?)
        # If multiple entities are mentioned, find scenes where they appear together OR in sequence
        if entities and len(entities) >= 2:
            results.extend(self._find_interaction(entities))

        # Strategy 2: Action Search
        if actions:
            for action in actions:
                results.extend(self._find_action(action))

        # Strategy 3: Text Search (if query looks like text match)
        # TODO: Detect if query is asking for text "Sign that says..."

        return self._deduplicate(results)

    def _find_interaction(self, entities: List[str]) -> List[Dict[str, Any]]:
        """Find scenes where entities appear together."""
        # This assumes entities are resolved to Cluster IDs or Names
        # For now, fuzzy match on names if we don't have IDs

        # Cypher: Find scenes containing Person A and Person B
        cypher = """
        MATCH (s:Scene)
        WHERE 
            EXISTS {
                MATCH (p1:Person)-[:APPEARED_IN]->(s)
                WHERE p1.id CONTAINS $e1 OR p1.cluster_id = $e1_int
            } AND EXISTS {
                MATCH (p2:Person)-[:APPEARED_IN]->(s)
                WHERE p2.id CONTAINS $e2 OR p2.cluster_id = $e2_int
            }
        RETURN s.id as id, s.start_time as start, s.end_time as end, s.description as description, 1.0 as score
        LIMIT 20
        """

        # Naive: just take first 2 entities
        e1 = entities[0]
        e2 = entities[1]

        # Try to cast to int if they are cluster IDs
        e1_int = int(e1) if str(e1).isdigit() else -1
        e2_int = int(e2) if str(e2).isdigit() else -1

        try:
            return self.store.query(cypher, {
                "e1": str(e1), "e1_int": e1_int,
                "e2": str(e2), "e2_int": e2_int
            })
        except Exception as e:
            log.error(f"[GraphSearch] Interaction search failed: {e}")
            return []

    def _find_action(self, action: str) -> List[Dict[str, Any]]:
        """Find scenes depicting an action."""
        cypher = """
        MATCH (s:Scene)-[:DEPICTS]->(a:Action)
        WHERE a.short_desc CONTAINS $action OR a.description CONTAINS $action
        RETURN s.id as id, s.start_time as start, s.end_time as end, s.description as description, 0.8 as score
        LIMIT 20
        """
        return self.store.query(cypher, {"action": action.lower()})

    def _deduplicate(self, results: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        seen = set()
        unique = []
        for r in results:
            rid = r.get("id")
            if rid not in seen:
                seen.add(rid)
                unique.append(r)
        return unique
