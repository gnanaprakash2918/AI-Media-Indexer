"""Pytest configuration for root."""

import sys
from unittest.mock import MagicMock

# Create a mock for torch because the real one hangs on import in this environment
mock_torch = MagicMock()
mock_torch.cuda.is_available.return_value = False
mock_torch.backends.mps.is_available.return_value = False
mock_torch.cuda.device_count.return_value = 0
mock_torch.__version__ = "2.0.0"

# Apply the mock to sys.modules BEFORE any other imports happen
sys.modules["torch"] = mock_torch

# sys.path hack for relative imports if needed
import os  # noqa: E402

sys.path.append(os.getcwd())
