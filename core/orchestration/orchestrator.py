"""LLM-Based Multi-Agent Orchestrator.

Prompts loaded from external files.
"""

import json
from typing import TYPE_CHECKING, Any

from core.agent.cards import (
    AUDIO_AGENT_TOOLS,
    SEARCH_AGENT_TOOLS,
    VISION_AGENT_TOOLS,
    get_all_tool_schemas,
)
from core.utils.logger import log
from core.utils.prompt_loader import load_prompt

if TYPE_CHECKING:
    from llm.interface import LLMInterface

ROUTER_PROMPT = load_prompt("orchestrator_routing")


class MultiAgentOrchestrator:
    """Routes queries to appropriate agents using LLM-based reasoning."""

    def __init__(self, llm: LLMInterface | None = None) -> None:
        """Initializes the orchestrator with an LLM and tool schemas.

        Args:
            llm: Optional LLM interface for reasoning and routing.
        """
        from llm.factory import LLMFactory

        self.llm = llm or LLMFactory.get_default_llm()
        self.tool_schemas = get_all_tool_schemas()
        self._tool_map = self._build_tool_map()

    def _build_tool_map(self) -> dict[str, str]:
        """Maps individual tool names to their parent agent types.

        Returns:
            A dictionary mapping tool names to 'vision', 'audio', or 'search'.
        """
        mapping = {}
        for t in VISION_AGENT_TOOLS:
            mapping[t.name] = "vision"
        for t in AUDIO_AGENT_TOOLS:
            mapping[t.name] = "audio"
        for t in SEARCH_AGENT_TOOLS:
            mapping[t.name] = "search"
        return mapping

    async def route(self, query: str) -> dict[str, Any]:
        """Routes a natural language query to the most appropriate agent.

        Uses LLM-based reasoning to analyze the query and select a tool
        from the available agent capability cards. Falls back to rule-based
        routing if the LLM is unavailable or fails.

        Args:
            query: The user's natural language query.

        Returns:
            A dictionary containing the selected 'agent', 'tool',
            and 'parameters'.
        """
        if self.llm is None:
            return self._fallback_route(query)

        prompt = ROUTER_PROMPT.format(
            tools_json=json.dumps(self.tool_schemas, indent=2), query=query
        )

        try:
            response = await self.llm.generate(prompt)
            result = json.loads(response)
            agent_type = self._tool_map.get(result.get("tool", ""), "search")
            log(f"[Orchestrator] Routed to {agent_type}: {result.get('tool')}")
            return {"agent": agent_type, **result}
        except Exception as e:
            log(f"[Orchestrator] LLM routing failed: {e}")
            return self._fallback_route(query)

    def _fallback_route(self, query: str) -> dict[str, Any]:
        """Provides a simplified rule-based routing when LLM is unavailable.

        Args:
            query: The user's natural language query.

        Returns:
            A dictionary containing the selected agent and tool.
        """
        q = query.lower()

        if any(
            w in q
            for w in ["transcribe", "speech", "audio", "speaker", "dialogue"]
        ):
            return {
                "agent": "audio",
                "tool": "transcribe_audio",
                "parameters": {"audio_path": ""},
                "reasoning": "Audio-related query",
            }

        if any(
            w in q for w in ["face", "person", "detect", "track", "segment"]
        ):
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
        """Orchestrates the routing and execution of a user query.

        Identifies the target agent and invokes the corresponding tool
        asynchronously.

        Args:
            query: The user's natural language query.

        Returns:
            A dictionary containing the routing information and execution results.
        """
        route_result = await self.route(query)
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
            from core.processing.vision import VisionAnalyzer

            # analyzer = VisionAnalyzer()
            # If "describe_image" or similar tool is requested, we might need a file path
            # However, the vision agent serves mostly for VQA on frames or scene analysis
            # For now, we'll map "analyze_frame" to VLM description if a frame path is provided

            # Simple implementation: Execute specific vision tool if mapped
            if tool_name == "analyze_frame":
                # In a real scenario, we'd need an image input.
                # This might be for re-analysis or specific detailed queries.
                # For the orchestrator to be effective, it needs context (which video/frame).
                # Temporarily, we return a structured response directing the user to the specific API
                return {
                    "route": route_result,
                    "results": {
                        "action": "visual_analysis",
                        "status": "ready",
                        "note": "Vision Agent ready to analyze specific frames via /api/vision/analyze",
                    },
                }

            return {
                "route": route_result,
                "results": [],
                "message": "Vision agent logic connected but requires specific frame context.",
            }

        elif agent_type == "audio":
            if tool_name == "transcribe_audio":
                # Check for context - we need a file path
                audio_path = params.get("audio_path") or params.get("path")
                if not audio_path:
                    return {
                        "route": route_result,
                        "results": {
                            "action": "audio_analysis",
                            "status": "waiting_input",
                            "note": "Please provide an audio/video file path to transcribe.",
                        },
                    }

                # In a real agentic loop, we would trigger the transcription job here
                # or return a function call to the client
                return {
                    "route": route_result,
                    "results": {
                        "action": "transcribe",
                        "target": audio_path,
                        "status": "ready",
                    },
                }

            return {
                "route": route_result,
                "results": [],
                "message": "Audio agent logic connected.",
            }

        return {"route": route_result, "results": []}


# Singleton instance
_orchestrator: MultiAgentOrchestrator | None = None


def get_orchestrator(llm: LLMInterface | None = None) -> MultiAgentOrchestrator:
    """Retrieves the singleton instance of the MultiAgentOrchestrator.

    Args:
        llm: Optional LLM interface override.

    Returns:
        The initialized MultiAgentOrchestrator instance.
    """
    global _orchestrator
    if _orchestrator is None:
        _orchestrator = MultiAgentOrchestrator(llm=llm)
    return _orchestrator
