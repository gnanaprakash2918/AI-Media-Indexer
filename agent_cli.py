"""Command-line Agent bridge between Ollama and the MediaIndexer MCP server.

This script:

1. Starts an MCP client over stdio to talk to your server:
   uv run python -m core.agent.server

2. Discovers available tools (search_media, ingest_media, ...).

3. Exposes them to an Ollama model via function-calling tools.

4. Runs an interactive REPL where the model can decide when to call tools,
   and the tool results are fed back into the model for a final answer.
"""

from __future__ import annotations

import asyncio
import json
import sys
from typing import Any, dict, list

import ollama
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client
from mcp.types import TextContent

from .config import settings

SERVER_PARAMS = StdioServerParameters(
    command="uv",
    args=["run", "python", "-m", "core.agent.server"],
    env=None,
)

MODEL_NAME = settings.agent_model


def _print_banner() -> None:
    """Print a small banner for the CLI."""
    print(f"\nMedia Agent (Ollama: {MODEL_NAME})")
    print("Type a question, or 'exit' / 'q' to quit.")


def _extract_text_from_content(contents: list[Any]) -> str:
    """Extract and concatenate text from an MCP tool result content list."""
    chunks: list[str] = []
    for item in contents:
        # FastMCP tool results are usually TextContent items
        if isinstance(item, TextContent):
            chunks.append(item.text)
        else:
            chunks.append(str(item))
    return "\n".join(chunks)


def _build_ollama_tools(tools: Any) -> list[dict[str, Any]]:
    """Convert MCP tools from list_tools() into Ollama function tool specs."""
    ollama_tools: list[dict[str, Any]] = []
    for t in tools:
        # t.inputSchema is already a JSON schema dict from FastMCP
        params_schema = t.inputSchema if isinstance(t.inputSchema, dict) else {}

        ollama_tools.append(
            {
                "type": "function",
                "function": {
                    "name": t.name,
                    "description": t.description or "",
                    "parameters": params_schema,
                },
            }
        )
    return ollama_tools


async def run_agent() -> None:
    """Main entrypoint: connect to MCP server and run interactive chat loop."""
    _print_banner()

    async with stdio_client(SERVER_PARAMS) as (read, write):
        async with ClientSession(read, write) as session:
            # 1. Initialize MCP session
            await session.initialize()

            # 2. Discover tools from the server
            tools_response = await session.list_tools()
            mcp_tools = tools_response.tools

            if not mcp_tools:
                print("[ERROR] No tools reported by MCP server. Exiting.")
                return

            print("[System] Connected to MCP server.")
            print("Available tools:")
            for t in mcp_tools:
                print(f"  - {t.name}: {t.description}")

            # 3. Build Ollama tools list
            ollama_tools = _build_ollama_tools(mcp_tools)

            # Conversation history for Ollama
            history: list[dict[str, Any]] = []

            # 4. Interactive loop
            while True:
                try:
                    user_msg = input("\nYou: ").strip()
                except (EOFError, KeyboardInterrupt):
                    print("\n[System] Goodbye.")
                    break

                if not user_msg:
                    continue
                if user_msg.lower() in {"q", "quit", "exit"}:
                    print("[System] Exiting.")
                    break

                # Add user message to history
                history.append({"role": "user", "content": user_msg})

                print("Agent (thinking)...")
                response = ollama.chat(
                    model=MODEL_NAME,
                    messages=history,
                    tools=ollama_tools,
                )

                msg = response.get("message", {})
                history.append(msg)

                tool_calls = msg.get("tool_calls") or []

                # Model wants to call tools
                if tool_calls:
                    for tool_call in tool_calls:
                        fn = tool_call.get("function", {})
                        fn_name: str = fn.get("name", "")
                        raw_args = fn.get("arguments", {}) or {}

                        # Parse arguments
                        if isinstance(raw_args, str):
                            try:
                                fn_args = json.loads(raw_args)
                            except json.JSONDecodeError:
                                print(
                                    f"[WARN] Tool '{fn_name}' arguments"
                                    "  were not valid JSON."
                                    "Calling with empty arguments."
                                )
                                fn_args = {}
                        else:
                            fn_args = raw_args

                        print(f"[Tool] Calling '{fn_name}' with args: {fn_args}")

                        # Call the MCP tool
                        try:
                            result = await session.call_tool(fn_name, fn_args)
                        except Exception as exc:  # noqa: BLE001
                            print(f"[ERROR] Tool '{fn_name}' failed: {exc}")
                            tool_output_text = f"Tool '{fn_name}' failed: {exc}"
                        else:
                            tool_output_text = _extract_text_from_content(
                                result.content
                            )
                            print(
                                f"[Tool] Result (truncated): {tool_output_text[:200]}"
                            )

                        # Feed tool result back to the model
                        history.append(
                            {
                                "role": "tool",
                                "name": fn_name,
                                "content": tool_output_text,
                            }
                        )

                    final_resp = ollama.chat(model=MODEL_NAME, messages=history)
                    final_msg = final_resp.get("message", {})
                    history.append(final_msg)
                    print(f"\nAgent: {final_msg.get('content', '').strip()}")
                    continue

                # No tool calls, just answer
                agent_text = msg.get("content", "").strip()
                print(f"\nAgent: {agent_text}")


def main() -> None:
    """Synchronous entrypoint wrapper for asyncio.run."""
    try:
        asyncio.run(run_agent())
    except KeyboardInterrupt:
        print("\n[System] Interrupted. Bye.")
        sys.exit(0)


if __name__ == "__main__":
    main()
