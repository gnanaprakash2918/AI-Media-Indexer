"""Logging helpers for AI-Media-Indexer.

This module provides a small `log` function that writes to stderr instead of
stdout. This is critical when running as an MCP server over stdio, because
stdout is reserved for JSON-RPC messages.
"""

from __future__ import annotations

import sys
from typing import Any


def log(*args: Any, **kwargs: Any) -> None:
    """Thread-safe, Unicode-safe logger for MCP Servers.

    Redirects to stderr and forces UTF-8 encoding to prevent Windows 'charmap' crashes.
    """
    # 1. Force stderr (Critical for MCP)
    kwargs["file"] = sys.stderr

    # 2. Convert to string
    msg = " ".join(str(arg) for arg in args)

    # 3. Write bytes directly to stderr buffer to bypass Windows console encoding issues
    try:
        # Add newline since print() usually does
        encoded_msg = (msg + "\n").encode("utf-8", errors="replace")
        sys.stderr.buffer.write(encoded_msg)
        sys.stderr.buffer.flush()
    except Exception:
        # If accessing buffer fails (rare), fall back to safe ASCII string
        safe_msg = msg.encode("ascii", errors="replace").decode("ascii")
        print(safe_msg, **kwargs)
