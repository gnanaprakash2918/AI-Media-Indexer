"""Verification script for the MCP server connectivity."""

import asyncio
import os
import sys

# Add project root to path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from mcp import ClientSession, StdioServerParameters  # noqa: E402
from mcp.client.stdio import stdio_client  # noqa: E402


async def verify_mcp_server():
    """Verify MCP server by connecting and listing tools."""
    print("🚀 Starting MCP Verification...")

    # Path to server script
    server_script = os.path.join(project_root, "core", "agent", "server.py")

    env = os.environ.copy()
    # Ensure project root is in PYTHONPATH for the subprocess
    env["PYTHONPATH"] = project_root + os.pathsep + env.get("PYTHONPATH", "")

    server_params = StdioServerParameters(
        command=sys.executable, args=[server_script], env=env
    )

    try:
        async with stdio_client(server_params) as (read, write):
            async with ClientSession(read, write) as session:
                # Initialize
                await session.initialize()
                print("✅ MCP Connection Established")

                # List Tools
                tools = await session.list_tools()
                print(f"🛠️  Found {len(tools.tools)} tools:")
                for tool in tools.tools:
                    print(
                        f"  - {tool.name}: {(tool.description or '')[:50]}..."
                    )

                # Check for required tools
                required = [
                    "query_video_rag",
                    "get_video_summary",
                    "enrich_identity",
                    "search_media",
                ]
                missing = [
                    t
                    for t in required
                    if not any(x.name == t for x in tools.tools)
                ]

                if missing:
                    print(f"❌ Missing expected tools: {missing}")
                    sys.exit(1)
                else:
                    print("✅ All required Phase 10 tools present.")

                # Done
                print("✅ Verification Complete!")

    except Exception as e:
        print(f"❌ Verification Failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    asyncio.run(verify_mcp_server())
