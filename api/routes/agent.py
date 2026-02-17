"""Agent API routes for LLM interaction and status checks.

Prompts loaded from external files.
"""

import time
from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException

from api.deps import get_pipeline
from api.schemas import AgentChatRequest
from core.ingestion.pipeline import IngestionPipeline
from core.utils.logger import logger
from core.utils.prompt_loader import load_prompt

router = APIRouter()


@router.get("/agent/status")
async def get_agent_status() -> dict:
    """Retrieves the AI agent's system status and its available tool capabilities.

    Checks the connectivity to the local Ollama instance and returns a list
    of tool schemas that the agent can currently utilize.

    Returns:
        A dictionary containing the 'active' status, Ollama connection info,
        and a list of available tool definitions.
    """
    """Get agent system status and available tools."""
    logger.info("[Agent] Status check requested")

    available_tools = [
        {
            "name": "agentic_search",
            "description": "FAANG-level search with LLM query expansion and identity resolution",
            "parameters": ["query", "limit", "use_expansion"],
        },
        {
            "name": "query_video_rag",
            "description": "Cognitive VideoRAG with query decomposition and answer generation",
            "parameters": ["query", "limit"],
        },
        {
            "name": "get_video_summary",
            "description": "Get hierarchical L1/L2 summaries for a video",
            "parameters": ["video_path", "force_regenerate"],
        },
        {
            "name": "search_hybrid",
            "description": "Hybrid vector + keyword search with identity boosting",
            "parameters": ["query", "video_path", "limit"],
        },
    ]

    ollama_status = "unknown"
    try:
        import ollama

        models = ollama.list()
        ollama_status = f"connected ({len(models.get('models', []))} models)"
    except Exception as e:
        ollama_status = f"error: {e}"

    return {
        "status": "active",
        "ollama": ollama_status,
        "tools": available_tools,
        "default_model": "llama3.2:3b",
    }


@router.post("/agent/chat")
async def agent_chat(
    request: AgentChatRequest,
    pipeline: Annotated[IngestionPipeline, Depends(get_pipeline)],
) -> dict:
    """Executes a multi-turn chat interaction with the AI agent.

    Processes the user's message using an LLM (Ollama) and orchestrates tool
    usage if requested. Log messages and tool executions are recorded for
    observability.

    Args:
        request: The chat request containing the user message and settings.
        pipeline: The initialized ingestion pipeline for context.

    Returns:
        A dictionary containing the agent's textual response and tool usage metadata.

    Raises:
        HTTPException: If the LLM interaction or tool execution fails.
    """
    """Send a message to the AI agent and get a response with tool usage logging."""
    logger.info(f"[Agent] RECV: '{request.message[:100]}...'")

    try:
        import ollama

        system_prompt = load_prompt("agent_chat_system")

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": request.message},
        ]

        if not request.use_tools:
            response = ollama.chat(model=request.model, messages=messages)
            return {
                "response": response["message"]["content"],
                "tools_used": [],
                "duration": 0,
            }

        # Real Orchestrator Logic
        from core.orchestration.orchestrator import get_orchestrator

        start_time = time.time()
        orchestrator = get_orchestrator()

        # Execute query through orchestrator
        # The orchestrator uses its own LLM for routing
        # The result includes the tool execution and result
        execution_result = await orchestrator.execute(request.message)

        duration = time.time() - start_time

        # Format response
        route = execution_result.get("route", {})
        results = execution_result.get("results")

        # Generate a final answer based on the tool result
        tool_name = route.get("tool", "unknown")
        agent_type = route.get("agent", "unknown")

        final_answer_prompt = f"""
        User Query: {request.message}
        Tool Used: {tool_name} ({agent_type})
        Tool Result: {results}
        
        Based on the tool result, answer the user's query gracefully.
        """

        final_response = ollama.chat(
            model=request.model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": final_answer_prompt},
            ],
        )

        return {
            "response": final_response["message"]["content"],
            "tools_used": [
                {"name": tool_name, "agent": agent_type, "result": results}
            ],
            "duration": duration,
        }

    except Exception as e:
        logger.error(f"[Agent] Chat failed: {e}")
        raise HTTPException(status_code=500, detail="Internal server error") from e
