"""A2A request handler for the Media Indexer agent.

This handler embeds an Ollama-backed LLM as the "brain" and uses the
local ingestion & search tools as its "hands". Incoming A2A messages
are mapped to a single LLM turn, optionally with tool calls.
"""

from __future__ import annotations

import json
import os
import uuid
from typing import Any

import ollama
from a2a.server.context import ServerCallContext
from a2a.server.request_handlers import RequestHandler
from a2a.types import (
    DeleteTaskPushNotificationConfigParams,
    GetTaskPushNotificationConfigParams,
    ListTaskPushNotificationConfigParams,
    Message,
    MessageSendParams,
    Part,
    Role,
    Task,
    TaskIdParams,
    TaskPushNotificationConfig,
    TaskQueryParams,
    TaskState,
    TaskStatus,
    TextPart,
    UnsupportedOperationError,
)
from a2a.utils.errors import ServerError
from loguru import logger

from core.agent.server import (
    agentic_search,
    get_video_summary,
    ingest_media,
    query_video_rag,
    scenelet_search,
    search_media,
)

DEFAULT_MODEL = os.getenv("MEDIA_AGENT_MODEL", "llama3.2:3b")


def _build_tool_schemas() -> list[dict[str, Any]]:
    """Return Ollama tool schemas for search and ingest.

    These schemas mirror the MCP tool signatures so that the model
    can reason about when and how to call them.
    """
    return [
        {
            "type": "function",
            "function": {
                "name": "search_media",
                "description": (
                    "Search the indexed media library for scenes or dialogue "
                    "that match a natural-language description."
                ),
                "parameters": {
                    "type": "object",
                    "properties": {
                        "query": {
                            "type": "string",
                            "description": (
                                "Natural language search query, e.g. "
                                "'red car explosion at night'."
                            ),
                        },
                        "limit": {
                            "type": "integer",
                            "description": "Maximum number of matches to return.",
                            "minimum": 1,
                            "maximum": 50,
                            "default": 5,
                        },
                    },
                    "required": ["query"],
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "ingest_media",
                "description": (
                    "Ingest and index a video file from the local filesystem."
                ),
                "parameters": {
                    "type": "object",
                    "properties": {
                        "file_path": {
                            "type": "string",
                            "description": (
                                "Absolute path to the video file on disk."
                            ),
                        },
                        "media_type": {
                            "type": "string",
                            "description": (
                                "Optional media type hint: 'movie', 'tv', "
                                "'personal', or 'unknown'."
                            ),
                            "default": "unknown",
                        },
                    },
                    "required": ["file_path"],
                },
            },
        },
    ] + _sota_search_tools()


def _sota_search_tools() -> list[dict[str, Any]]:
    """Returns schemas for the new SOTA search tools."""
    return [
        {
            "type": "function",
            "function": {
                "name": "agentic_search",
                "description": (
                    "Advanced natural language search with query expansion and identity resolution. "
                    "Use this for finding specific objects, actions, clothing, or people."
                ),
                "parameters": {
                    "type": "object",
                    "properties": {
                        "query": {
                            "type": "string",
                            "description": "Search query.",
                        },
                        "limit": {"type": "integer", "default": 10},
                        "use_expansion": {"type": "boolean", "default": True},
                    },
                    "required": ["query"],
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "scenelet_search",
                "description": (
                    "Search for temporal action sequences (e.g. 'running then jumping'). "
                    "Returns time ranges."
                ),
                "parameters": {
                    "type": "object",
                    "properties": {
                        "query": {
                            "type": "string",
                            "description": "Action query.",
                        },
                        "limit": {"type": "integer", "default": 10},
                        "video_path": {
                            "type": "string",
                            "description": "Optional video filter.",
                        },
                    },
                    "required": ["query"],
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "get_video_summary",
                "description": "Get a hierarchical summary of a video's content (plot, events).",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "video_path": {
                            "type": "string",
                            "description": "Path to video.",
                        },
                        "force_regenerate": {
                            "type": "boolean",
                            "default": False,
                        },
                    },
                    "required": ["video_path"],
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "query_video_rag",
                "description": (
                    "Cognitive 'High IQ' search that answers questions using video content "
                    "and external knowledge. Use for 'Why/How' questions."
                ),
                "parameters": {
                    "type": "object",
                    "properties": {
                        "query": {
                            "type": "string",
                            "description": "Question or query.",
                        },
                        "limit": {"type": "integer", "default": 10},
                    },
                    "required": ["query"],
                },
            },
        },
    ]


class MediaAgentHandler(RequestHandler):
    """A2A RequestHandler that orchestrates Ollama + local tools.

    The control flow for each incoming message is:

    1. Extract the user's text from the A2A `Message`.
    2. Ask the Ollama model, providing the tool schemas.
    3. If the model emits tool calls:
       * Execute `search_media`/`ingest_media` directly (async).
       * Feed tool results back into the model for a final answer.
    4. Wrap the final answer as a `Task` with state `completed`.
    """

    def __init__(self, model_name: str | None = None) -> None:
        """Initialize the handler.

        Args:
            model_name: Optional override for the Ollama model name.
                Defaults to the `MEDIA_AGENT_MODEL` environment variable
                or ``llama3.2:3b``.
        """
        self.model_name = model_name or DEFAULT_MODEL
        self.tools = _build_tool_schemas()
        logger.info("media_agent_handler.initialized", model=self.model_name)

    async def on_message_send(
        self,
        params: MessageSendParams,
        context: ServerCallContext | None = None,
    ) -> Task:
        """Handle an A2A `message.send` request."""
        # 1. Extract user text
        logger.info("DEBUG: Handler received request!")
        user_message: Message = params.message  # type: ignore[assignment]
        text_parts = [
            p.root.text
            for p in user_message.parts
            if isinstance(p.root, TextPart)
        ]
        user_text = "\n".join(text_parts).strip()

        logger.info(
            "media_agent_handler.received_message",
            context_id=getattr(user_message, "context_id", None),
            text_preview=user_text[:200],
        )

        if not user_text:
            reply_text = "I didn't receive any text to work with."
            return self._build_task(
                user_message=user_message,
                reply_text=reply_text,
            )

        # 2. First LLM call with tools
        history: list[dict[str, Any]] = [
            {"role": "user", "content": user_text},
        ]

        try:
            llm_response = ollama.chat(
                model=self.model_name,
                messages=history,
                tools=self.tools,
            )
        except Exception as exc:
            logger.exception("media_agent_handler.ollama_error", exc_info=exc)
            reply_text = f"The language model failed while processing your request: {exc}"
            return self._build_task(
                user_message=user_message,
                reply_text=reply_text,
            )

        msg = llm_response.get("message", {})
        history.append(msg)

        tool_calls = msg.get("tool_calls") or []

        # 3. If there are tool calls, execute them
        if tool_calls:
            logger.info(
                "media_agent_handler.tool_calls_detected",
                count=len(tool_calls),
            )

            for call in tool_calls:
                fn = call.get("function", {})
                fn_name: str = fn.get("name", "")
                raw_args = fn.get("arguments") or {}

                # Some models send arguments as JSON string
                if isinstance(raw_args, str):
                    try:
                        args = json.loads(raw_args)
                    except json.JSONDecodeError:
                        args = {}
                else:
                    args = raw_args

                logger.info(
                    "media_agent_handler.tool_call",
                    name=fn_name,
                    args=args,
                )

                tool_result_text = await self._execute_tool(fn_name, args)

                # Feed tool result back into the conversation
                history.append(
                    {
                        "role": "tool",
                        "content": tool_result_text,
                        "name": fn_name,
                    },
                )

            # 4. Second LLM call to generate final answer
            try:
                final = ollama.chat(
                    model=self.model_name,
                    messages=history,
                )
                reply_text = final["message"]["content"]
            except Exception as exc:
                logger.exception(
                    "media_agent_handler.ollama_followup_error",
                    exc_info=exc,
                )
                reply_text = (
                    "I executed the requested tools, but failed while "
                    f"forming the final answer: {exc}"
                )
        else:
            reply_text = msg.get("content", "")

        return self._build_task(
            user_message=user_message,
            reply_text=reply_text,
        )

    async def _execute_tool(self, name: str, args: dict[str, Any]) -> str:
        """Execute a local tool by name and return a text summary."""
        if name == "search_media":
            query = args.get("query", "")
            limit = int(args.get("limit", 5))
            try:
                search_result = await search_media(query=query, limit=limit)
                # `search_result` is a pydantic model (SearchResponse)
                data = search_result.model_dump()
                visual = data.get("visual_matches", [])
                dialogue = data.get("dialogue_matches", [])

                lines: list[str] = []
                lines.append(f"Search results for: {query!r}")
                lines.append(f"Visual matches: {len(visual)}")
                for hit in visual[:5]:
                    lines.append(
                        f"- [V] {hit.get('time')} {hit.get('file')}: "
                        f"{hit.get('content')}",
                    )
                lines.append(f"Dialogue matches: {len(dialogue)}")
                for hit in dialogue[:5]:
                    lines.append(
                        f"- [D] {hit.get('time')} {hit.get('file')}: "
                        f"{hit.get('content')}",
                    )

                return "\n".join(lines)
            except Exception as exc:
                logger.exception(
                    "media_agent_handler.search_error", exc_info=exc
                )
                return f"search_media failed with error: {exc}"

        if name == "ingest_media":
            file_path = args.get("file_path", "")
            media_type = args.get("media_type", "unknown")
            try:
                ingest_result = await ingest_media(
                    file_path=file_path,
                    media_type=media_type,
                )
                # IngestResponse is a pydantic model
                return (
                    f"Ingestion result for {ingest_result.file_path}: "
                    f"{ingest_result.message} "
                    f"(media_type_hint={ingest_result.media_type_hint})"
                )
            except Exception as exc:
                logger.exception(
                    "media_agent_handler.ingest_error", exc_info=exc
                )
                return f"ingest_media failed with error: {exc}"

        if name == "agentic_search":
            try:
                res = await agentic_search(
                    query=args.get("query", ""),
                    limit=int(args.get("limit", 10)),
                    use_expansion=args.get("use_expansion", True),
                )
                # Format results broadly
                count = res.get("result_count", 0)
                reasoning = res.get("reasoning_chain", {})
                hits = res.get("results", [])

                lines = [f"Found {count} results using agentic search."]
                lines.append(f"Reasoning: {reasoning}")
                for h in hits[:5]:
                    lines.append(
                        f"- {h.get('score', 0):.2f} | {h.get('video_name')}: {h.get('description')}"
                    )
                return "\n".join(lines)
            except Exception as e:
                return f"agentic_search failed: {e}"

        if name == "scenelet_search":
            try:
                res = await scenelet_search(
                    query=args.get("query", ""),
                    limit=int(args.get("limit", 10)),
                    video_path=args.get("video_path"),
                )
                hits = res.get("results", [])
                lines = [f"Found {len(hits)} scenelets."]
                for h in hits[:5]:
                    lines.append(
                        f"- {h.get('start_time'):.1f}s-{h.get('end_time'):.1f}s: {h.get('match_explanation')}"
                    )
                return "\n".join(lines)
            except Exception as e:
                return f"scenelet_search failed: {e}"

        if name == "get_video_summary":
            try:
                res = await get_video_summary(
                    video_path=args.get("video_path"),
                    force_regenerate=args.get("force_regenerate", False),
                )
                l1 = res.get("l1_summary", "No summary available.")
                return f"Video Summary:\n{l1}"
            except Exception as e:
                return f"Summary failed: {e}"

        if name == "query_video_rag":
            try:
                res = await query_video_rag(
                    query=args.get("query", ""),
                    limit=int(args.get("limit", 10)),
                )
                # If RAG generated an answer, return it
                if res.get("answer"):
                    return f"Answer: {res.get('answer')}"

                # Otherwise list results
                hits = res.get("results", [])
                return f"Found {len(hits)} relevant segments for RAG context."
            except Exception as e:
                return f"RAG failed: {e}"

        logger.warning("media_agent_handler.unknown_tool", name=name)
        return f"Unknown tool: {name}"

    @staticmethod
    def _build_task(user_message: Message, reply_text: str) -> Task:
        """Construct a completed Task with the given reply text."""
        context_id = getattr(user_message, "context_id", None) or "ctx-1"
        task_id = f"task-{uuid.uuid4()}"

        reply_message = Message(
            role=Role.agent,
            message_id=f"msg-{uuid.uuid4()}",
            parts=[Part(root=TextPart(text=reply_text))],
            context_id=context_id,
        )

        status = TaskStatus(
            state=TaskState.completed,
            message=reply_message,
        )

        return Task(
            id=task_id,
            context_id=context_id,
            status=status,
            kind="task",
        )

    async def on_get_task(
        self, params: TaskQueryParams, context: ServerCallContext | None = None
    ) -> Task | None:
        """Retrieves a specific task's status.

        Note: Task history is currently not persisted.

        Returns:
            Always None in this implementation.
        """
        return None

    async def on_cancel_task(
        self, params: TaskIdParams, context: ServerCallContext | None = None
    ) -> Task | None:
        """Cancels a running task.

        Raises:
            ServerError: Implementation not supported.
        """
        raise ServerError(
            error=UnsupportedOperationError(
                message="Cancellation not supported yet"
            )
        )

    async def on_message_send_stream(
        self,
        params: MessageSendParams,
        context: ServerCallContext | None = None,
    ):
        """Streaming version of message send.

        Raises:
            ServerError: Implementation not supported.
        """
        # We will implement this in Sprint 8 for streaming
        raise ServerError(
            error=UnsupportedOperationError(
                message="Streaming not supported yet"
            )
        )
        yield  # Required for generator return type

    async def on_set_task_push_notification_config(
        self,
        params: TaskPushNotificationConfig,
        context: ServerCallContext | None = None,
    ) -> TaskPushNotificationConfig:
        """Sets push notification configuration for a task.

        Raises:
            ServerError: Implementation not supported.
        """
        raise ServerError(error=UnsupportedOperationError())

    async def on_get_task_push_notification_config(
        self,
        params: TaskIdParams | GetTaskPushNotificationConfigParams,
        context: ServerCallContext | None = None,
    ) -> TaskPushNotificationConfig:
        """Retrieves push notification configuration for a task.

        Raises:
            ServerError: Implementation not supported.
        """
        raise ServerError(error=UnsupportedOperationError())

    async def on_resubscribe_to_task(
        self, params: TaskIdParams, context: ServerCallContext | None = None
    ):
        """Resubscribes to task updates.

        Raises:
            ServerError: Implementation not supported.
        """
        raise ServerError(error=UnsupportedOperationError())
        yield

    async def on_list_task_push_notification_config(
        self,
        params: ListTaskPushNotificationConfigParams,
        context: ServerCallContext | None = None,
    ) -> list[TaskPushNotificationConfig]:
        """Lists push notification configurations.

        Returns:
            Always an empty list.
        """
        return []

    async def on_delete_task_push_notification_config(
        self,
        params: DeleteTaskPushNotificationConfigParams,
        context: ServerCallContext | None = None,
    ) -> None:
        """Deletes push notification configuration."""
        pass
