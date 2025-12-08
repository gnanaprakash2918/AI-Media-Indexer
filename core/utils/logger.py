"""Logging helpers for AI-Media-Indexer.

This module provides a small `log` function that writes to stderr instead of
stdout. This is critical when running as an MCP server over stdio, because
stdout is reserved for JSON-RPC messages.
"""

from __future__ import annotations

import sys
from typing import Any


def log(*args: Any, **kwargs: Any) -> None:
    """Write a log message to stderr.

    Args:
        *args: Positional arguments passed to :func:`print`.
        **kwargs: Keyword arguments passed to :func:`print`. The ``file``
            parameter is always forced to :data:`sys.stderr`.
    """
    # Ensure we never write logs to stdout (which MCP uses for JSON-RPC).
    kwargs.setdefault("file", sys.stderr)
    print(*args, **kwargs)
