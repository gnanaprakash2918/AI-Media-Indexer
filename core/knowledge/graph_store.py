"""Graph Store: Neo4j Interface for the Knowledge Graph.

Handles connection to Neo4j and execution of Cypher queries.
Part of the GraphRAG architecture.
"""

import os
from typing import Any, Dict, List, Optional

from neo4j import GraphDatabase

from core.utils.logger import get_logger

log = get_logger(__name__)


class GraphStore:
    """Interface for Neo4j Graph Database."""

    def __init__(self, uri: str = "bolt://localhost:7687", auth: tuple = ("neo4j", "password")):
        """Initialize Neo4j driver.
        
        Args:
            uri: Bolt URI (e.g. bolt://localhost:7687)
            auth: Tuple of (username, password)
        """
        # Allow env override
        self.uri = os.getenv("NEO4J_URI", uri)
        user = os.getenv("NEO4J_USER", auth[0])
        password = os.getenv("NEO4J_PASSWORD", auth[1])

        try:
            self.driver = GraphDatabase.driver(self.uri, auth=(user, password))
            self.verify_connectivity()
            log.info(f"[GraphStore] Connected to Neo4j at {self.uri}")
        except Exception as e:
            log.error(f"[GraphStore] Failed to connect to Neo4j: {e}")
            self.driver = None

    def verify_connectivity(self):
        """Check if connection is alive."""
        if self.driver:
            self.driver.verify_connectivity()

    def close(self):
        """Close the driver."""
        if self.driver:
            self.driver.close()
            log.info("[GraphStore] Connection closed.")

    def query(self, cypher: str, params: Dict[str, Any] = None) -> List[Dict[str, Any]]:
        """Execute a Cypher query and return list of dict results."""
        if not self.driver:
            log.warning("[GraphStore] Driver not initialized, skipping query.")
            return []

        if params is None:
            params = {}

        try:
            results = self.driver.execute_query(cypher, parameters_=params)
            # Neo4j driver returns EagerResult object in recent versions
            # results.records is the list of records
            records = results.records
            return [r.data() for r in records]
        except Exception as e:
            log.error(f"[GraphStore] Query failed:\n{cypher}\nError: {e}")
            return []

    def wipe_db(self):
        """Dangerous: Wipe all data."""
        log.warning("[GraphStore] Wiping ALL data from Graph DB...")
        self.query("MATCH (n) DETACH DELETE n")

    def ensure_indices(self):
        """Create essential indices for performance."""
        indices = [
            "CREATE INDEX IF NOT EXISTS FOR (s:Scene) ON (s.id)",
            "CREATE INDEX IF NOT EXISTS FOR (p:Person) ON (p.name)",
            "CREATE INDEX IF NOT EXISTS FOR (v:Video) ON (v.path)",
        ]
        for idx in indices:
            self.query(idx)

# Singleton global instance
_GRAPH_STORE = None

def get_graph_store() -> Optional[GraphStore]:
    """Get or create singleton GraphStore."""
    global _GRAPH_STORE
    if _GRAPH_STORE is None:
        # Default credentials as per docker-compose
        try:
            _GRAPH_STORE = GraphStore()
        except Exception as e:
            log.error(f"[GraphStore] Could not initialize global store: {e}")
            return None
    return _GRAPH_STORE
