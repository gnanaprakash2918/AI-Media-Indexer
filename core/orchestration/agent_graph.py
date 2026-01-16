"""A2A Multi-Agent Orchestrator.

Routes complex user queries to specialized agents (Search, Vision, Audio)
using Semantic Routing (LLM-based decision making).

Prompts loaded from external files for easy customization.
"""

import json
import logging
from dataclasses import dataclass
from pathlib import Path

logger = logging.getLogger(__name__)

# Prompt file path
PROMPTS_DIR = Path(__file__).parent.parent.parent / "prompts"


def _load_prompt(name: str) -> str:
    """Load prompt from external file."""
    prompt_file = PROMPTS_DIR / f"{name}.txt"
    if prompt_file.exists():
        return prompt_file.read_text(encoding="utf-8")
    logger.warning(f"Prompt file not found: {prompt_file}")
    return ""


# Load router prompt from external file
ROUTER_PROMPT = _load_prompt("agent_routing")


@dataclass
class AgentResponse:
    """Represents the response from a routed agent."""

    agent_name: str = ""
    content: str = ""
    error: str = ""


class AgentCard:
    """Capability card for an agent."""

    def __init__(
        self, name: str, description: str, capabilities: list[str] | None = None
    ):
        """Initialize the agent card."""
        self.name = name
        self.description = description
        self.capabilities = capabilities or []


class MultiAgentOrchestrator:
    """Routes queries to appropriate agents using LLM-based semantic routing."""

    def __init__(self, llm=None):
        """Initializes the orchestrator.

        Args:
            llm: Optional LLM instance.
        """
        self._llm = llm
        self._llm_loaded = False
        self.agents: dict[str, AgentCard] = {}
        self._register_default_agents()

    def _register_default_agents(self):
        """Register the default agent cards."""
        self.register_agent(
            "search_agent",
            AgentCard(
                name="search_agent",
                description="Searches indexed media for specific moments, people, objects, actions. Handles complex queries like 'Prakash bowling at Brunswick'.",
                capabilities=[
                    "semantic_search",
                    "identity_filter",
                    "temporal_search",
                ],
            ),
        )
        self.register_agent(
            "vision_agent",
            AgentCard(
                name="vision_agent",
                description="Analyzes images/frames for objects, people, text, scene context using VLM.",
                capabilities=["frame_analysis", "object_detection", "ocr"],
            ),
        )
        self.register_agent(
            "audio_agent",
            AgentCard(
                name="audio_agent",
                description="Handles transcripts, speech, speaker identification, and dialogue search.",
                capabilities=[
                    "transcription",
                    "speaker_diarization",
                    "dialogue_search",
                ],
            ),
        )
        # NEW: Advanced query decomposition for extremely complex queries
        self.register_agent(
            "query_decomposition_agent",
            AgentCard(
                name="query_decomposition_agent",
                description=(
                    "Handles ultra-complex 200+ word queries with multiple constraints. "
                    "Decomposes queries into person, clothing, action, location, temporal, "
                    "audio aspects. Performs multi-vector search with late fusion. "
                    "Example: 'Prakash in blue t-shirt, John Jacobs spectacles, red/green "
                    "mismatched shoes bowling at Brunswick with last pin falling slowly'"
                ),
                capabilities=[
                    "multi_vector_search",
                    "query_decomposition",
                    "ner_extraction",
                    "temporal_reasoning",
                    "aspect_fusion",
                    "exclusion_filtering",
                ],
            ),
        )

    def register_agent(self, name: str, card: AgentCard):
        """Registers an agent with its capabilities card."""
        self.agents[name] = card
        logger.info(f"Registered Agent: {name}")

    def _ensure_llm_loaded(self):
        """Lazy load LLM."""
        if self._llm is None and not self._llm_loaded:
            try:
                from llm.factory import LLMFactory

                self._llm = LLMFactory.create_llm(provider="ollama")
                self._llm_loaded = True
            except Exception as e:
                logger.warning(f"Could not load LLM for routing: {e}")
                self._llm_loaded = True  # Don't retry

    def route_request_sync(self, user_query: str) -> AgentResponse:
        """Synchronous routing using rule-based fallback."""
        return self._fallback_route(user_query)

    async def route_request(self, user_query: str) -> AgentResponse:
        """Decides which agent handles the query using LLM."""
        if not self.agents:
            return AgentResponse(error="No agents registered.")

        self._ensure_llm_loaded()

        # Fallback to rule-based if no LLM
        if self._llm is None:
            return self._fallback_route(user_query)

        # Build Agent Descriptions for Prompt
        agents_desc = "\n".join(
            [
                f"- {name}: {card.description}"
                for name, card in self.agents.items()
            ]
        )

        try:
            response_text = await self._llm.generate(
                ROUTER_PROMPT.format(agents_desc=agents_desc, query=user_query)
            )
            # Clean JSON markdown
            response_text = (
                response_text.replace("```json", "").replace("```", "").strip()
            )
            routing_data = json.loads(response_text)

            selected_agent_name = routing_data.get("selected_agent")
            if selected_agent_name not in self.agents:
                logger.warning(
                    f"Router selected unknown agent: {selected_agent_name}"
                )
                return self._fallback_route(user_query)

            logger.info(
                f"Routing to {selected_agent_name}: {routing_data.get('reasoning', '')}"
            )

            return AgentResponse(
                agent_name=selected_agent_name,
                content=routing_data.get("refined_query", user_query),
            )
        except Exception as e:
            logger.error(f"Routing failed: {e}")
            return self._fallback_route(user_query)

    def _fallback_route(self, user_query: str) -> AgentResponse:
        """Rule-based fallback routing when LLM unavailable."""
        query_lower = user_query.lower()

        # Audio queries
        if any(
            w in query_lower
            for w in [
                "said",
                "say",
                "transcript",
                "speech",
                "subtitle",
                "dialogue",
            ]
        ):
            return AgentResponse(agent_name="audio_agent", content=user_query)

        # Vision queries (describe specific frame)
        if (
            any(w in query_lower for w in ["describe", "analyze", "explain"])
            and "frame" in query_lower
        ):
            return AgentResponse(agent_name="vision_agent", content=user_query)

        # Ultra-complex queries (>50 words or many constraints)
        word_count = len(user_query.split())
        constraint_keywords = [
            "wearing",
            "with",
            "while",
            "at",
            "on",
            "in",
            "excluding",
            "without",
            "followed by",
            "after",
            "before",
            "slowly",
            "quickly",
            "exactly",
            "ms",
            "seconds",
            "minutes",
            "dB",
            "BPM",
        ]
        constraint_count = sum(
            1 for kw in constraint_keywords if kw in query_lower
        )

        if word_count > 50 or constraint_count >= 5:
            logger.info(
                f"Routing to query_decomposition_agent: "
                f"{word_count} words, {constraint_count} constraints"
            )
            return AgentResponse(
                agent_name="query_decomposition_agent", content=user_query
            )

        # Default to search
        return AgentResponse(agent_name="search_agent", content=user_query)


# Singleton
_orchestrator: MultiAgentOrchestrator | None = None


def get_orchestrator() -> MultiAgentOrchestrator:
    """Retrieves the singleton instance of the MultiAgentOrchestrator.

    Returns:
        The initialized MultiAgentOrchestrator instance.
    """
    global _orchestrator
    if _orchestrator is None:
        _orchestrator = MultiAgentOrchestrator()
    return _orchestrator
