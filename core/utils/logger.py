"""Logging helpers for AI-Media-Indexer.

This module provides a small `log` function that writes to stderr instead of
stdout. This is critical when running as an MCP server over stdio, because
stdout is reserved for JSON-RPC messages.
"""

from __future__ import annotations

import sys
from typing import Any


def log(*args: Any, **kwargs: Any) -> None:
    """Logger that handles Unicode characters (like '｜') on Windowswithout crashing."""
    # Force output to stderr (MCP requirement)
    kwargs["file"] = sys.stderr

    # Convert all args to string
    msg = " ".join(str(arg) for arg in args)

    try:
        # Attempt to write directly
        print(msg, **kwargs)
    except UnicodeEncodeError:
        # Fallback: Replace unprintable characters with '?'
        # This works even if the console is strictly cp1252
        safe_msg = msg.encode("ascii", errors="replace").decode("ascii")
        print(safe_msg, **kwargs)
