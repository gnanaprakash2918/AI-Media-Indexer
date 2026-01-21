"""Windows compatibility layer for Unix-specific functionality.

This module should be imported early in the application startup to patch
missing Unix signals and other platform-specific features that third-party
libraries may expect.

Usage:
    import core.utils.platform_compat  # noqa: F401  (import for side effects)
"""

from __future__ import annotations

import signal
import sys

# Windows lacks certain Unix signals that some libraries (pyannote, pytorch-lightning) expect
if sys.platform == "win32":
    # SIGKILL (9): Used by pytorch-lightning's fault tolerance
    if not hasattr(signal, "SIGKILL"):
        signal.SIGKILL = 9
    
    # SIGTERM (15): Graceful termination - usually exists on Windows but ensure it
    if not hasattr(signal, "SIGTERM"):
        signal.SIGTERM = 15
    
    # SIGUSR1/SIGUSR2: User-defined signals, not available on Windows
    if not hasattr(signal, "SIGUSR1"):
        signal.SIGUSR1 = 10
    if not hasattr(signal, "SIGUSR2"):
        signal.SIGUSR2 = 12
