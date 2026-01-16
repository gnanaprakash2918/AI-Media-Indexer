"""Pytest configuration."""

import os
import sys

import pytest

project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_root not in sys.path:
    sys.path.insert(0, project_root)


@pytest.fixture(scope="session", autouse=True)
def setup_test_env():
    """Sets up environment variables for the entire test session."""
    os.environ["TEST_MODE"] = "true"
    os.environ["HIGH_PERFORMANCE_MODE"] = "false"
    os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
    os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"


@pytest.fixture
def mock_qdrant():
    """Mock Qdrant client for unit tests."""
    from unittest.mock import MagicMock

    mock_client = MagicMock()
    mock_client.get_collections.return_value = MagicMock(collections=[])
    mock_client.collection_exists.return_value = True
    return mock_client
