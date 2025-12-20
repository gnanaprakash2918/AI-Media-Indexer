import pytest
import asyncio
from pathlib import Path
from unittest.mock import MagicMock, patch

from core.utils.logger import setup_logger
from core.utils.retry import retry
from core.utils.resource import get_resource_usage
from core.processing.text_utils import parse_srt, _timestamp_to_seconds
from config import Settings

# --- Logger Tests ---

def test_logger_setup():
    # Just verify it doesn't crash
    setup_logger()

# --- Retry Utils Tests ---

@pytest.mark.asyncio
async def test_retry_success():
    call_count = 0
    async def task():
        nonlocal call_count
        call_count += 1
    await retry(task, retries=2, delay=0.01)
    assert call_count == 1

@pytest.mark.asyncio
async def test_retry_failure():
    call_count = 0
    async def fail_task():
        nonlocal call_count
        call_count += 1
        raise ValueError("Fail")
    with pytest.raises(ValueError):
        await retry(fail_task, retries=2, delay=0.01)
    assert call_count == 3

# --- Resource Utils Tests ---

def test_resource_usage():
    usage = get_resource_usage()
    assert "cpu_percent" in usage
    assert "memory_used_gb" in usage

# --- Text Utils Tests ---

def test_parse_srt(tmp_path):
    srt_content = "1\n00:00:01,000 --> 00:00:04,000\nHello"
    f = tmp_path / "test.srt"
    f.write_text(srt_content)
    segments = parse_srt(f)
    assert len(segments) == 1
    assert segments[0]["text"] == "Hello"

def test_timestamp_to_seconds():
    assert _timestamp_to_seconds("00:00:01.500") == 1.5
    assert _timestamp_to_seconds("invalid") == 0.0

# --- Config Tests ---

def test_config_defaults():
    settings = Settings()
    assert isinstance(settings.qdrant_port, int)
    assert settings.qdrant_host is not None

def test_config_env_override(monkeypatch):
    monkeypatch.setenv("QDRANT_HOST", "1.2.3.4")
    settings = Settings()
    assert settings.qdrant_host == "1.2.3.4"
