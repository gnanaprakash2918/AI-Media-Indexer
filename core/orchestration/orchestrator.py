"""LLM-Based Multi-Agent Orchestrator with A2A Protocol Compliance.

Uses LLM reasoning to route queries to appropriate agents based on their capability cards.
"""

import json
from typing import Any

from core.agent.cards import (
    AUDIO_AGENT_TOOLS,
    SEARCH_AGENT_TOOLS,
    VISION_AGENT_TOOLS,
    get_all_tool_schemas,
)
from core.utils.logger import log

ROUTER_PROMPT = """You are an AI orchestrator for a media indexing system. Given a user query, select the best tool(s) to execute.

Available tools:
{tools_json}

User query: {query}

Respond with JSON containing:
- "tool": tool name to call
- "parameters": dict of parameter values
- "reasoning": brief explanation

Example:
{{"tool": "search_scenes", "parameters": {{"query": "person bowling", "person": "Prakash"}}, "reasoning": "User wants to find Prakash bowling, using scene search with identity filter"}}
"""


class MultiAgentOrchestrator:
    """Routes queries to appropriate agents using LLM-based reasoning."""

    def __init__(self, llm=None):
        self.llm = llm
        self.tool_schemas = get_all_tool_schemas()
        self._tool_map = self._build_tool_map()

    def _build_tool_map(self) -> dict[str, str]:
        """Map tool names to agent types."""
        mapping = {}
        for t in VISION_AGENT_TOOLS:
            mapping[t.name] = "vision"
        for t in AUDIO_AGENT_TOOLS:
            mapping[t.name] = "audio"
        for t in SEARCH_AGENT_TOOLS:
            mapping[t.name] = "search"
        return mapping

    def route(self, query: str) -> dict[str, Any]:
        """Route query to appropriate agent using LLM reasoning."""
        if self.llm is None:
            return self._fallback_route(query)

        prompt = ROUTER_PROMPT.format(
            tools_json=json.dumps(self.tool_schemas, indent=2), query=query
        )

        try:
            response = self.llm.generate(prompt)
            result = json.loads(response)
            agent_type = self._tool_map.get(result.get("tool", ""), "search")
            log(f"[Orchestrator] Routed to {agent_type}: {result.get('tool')}")
            return {"agent": agent_type, **result}
        except Exception as e:
            log(f"[Orchestrator] LLM routing failed: {e}")
            return self._fallback_route(query)

    def _fallback_route(self, query: str) -> dict[str, Any]:
        """Rule-based fallback when LLM unavailable."""
        q = query.lower()

        if any(
            w in q for w in ["transcribe", "speech", "audio", "speaker", "dialogue"]
        ):
            return {
                "agent": "audio",
                "tool": "transcribe_audio",
                "parameters": {"audio_path": ""},
                "reasoning": "Audio-related query",
            }

        if any(w in q for w in ["face", "person", "detect", "track", "segment"]):
            return {
                "agent": "vision",
                "tool": "analyze_frame",
                "parameters": {},
                "reasoning": "Vision-related query",
            }

        # Default to search
        return {
            "agent": "search",
            "tool": "search_scenes",
            "parameters": {"query": query},
            "reasoning": "Default to scene search",
        }

    async def execute(self, query: str) -> dict[str, Any]:
        """Route and execute query through appropriate agent."""
        route_result = self.route(query)
        agent_type = route_result.get("agent", "search")
        tool_name = route_result.get("tool", "search_scenes")
        params = route_result.get("parameters", {})

        log(f"[Orchestrator] Executing {tool_name} on {agent_type} agent")

        if agent_type == "search":
            from core.retrieval.agentic_search import SearchAgent
            from core.storage.db import VectorDB

            db = VectorDB()
            agent = SearchAgent(db=db)
            results = await agent.search(params.get("query", query))
            return {"route": route_result, "results": results}

        elif agent_type == "vision":
            return {
                "route": route_result,
                "results": [],
                "message": "Vision agent execution placeholder",
            }

        elif agent_type == "audio":
            return {
                "route": route_result,
                "results": [],
                "message": "Audio agent execution placeholder",
            }

        return {"route": route_result, "results": []}


# Singleton instance
_orchestrator: MultiAgentOrchestrator | None = None


def get_orchestrator(llm=None) -> MultiAgentOrchestrator:
    global _orchestrator
    if _orchestrator is None:
        _orchestrator = MultiAgentOrchestrator(llm=llm)
    return _orchestrator
