import pytest
import asyncio
from pathlib import Path
from unittest.mock import MagicMock, patch

from core.utils.logger import setup_logger
from core.utils.retry import retry
from core.utils.resource import get_resource_usage
from core.processing.text_utils import parse_srt, _timestamp_to_seconds
from core.utils.resource import get_resource_usage
from core.processing.text_utils import parse_srt, _timestamp_to_seconds
from core.utils.frame_sampling import FrameSampler
from core.utils.progress import ProgressTracker
from core.utils.observability import start_trace, end_trace, start_span, end_span, trace_id_ctx
from core.utils.observe import observe
from config import Settings

def test_logger_setup():
    setup_logger()

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

def test_resource_usage():
    usage = get_resource_usage()
    assert "cpu_percent" in usage
    assert "memory_used_gb" in usage

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


def test_config_defaults():
    settings = Settings()
    assert isinstance(settings.qdrant_port, int)
    assert settings.qdrant_host is not None

def test_config_env_override(monkeypatch):
    monkeypatch.setenv("QDRANT_HOST", "1.2.3.4")
    settings = Settings()
    assert settings.qdrant_host == "1.2.3.4"

def test_frame_sampler():
    fs = FrameSampler(every_n=2)
    assert fs.should_sample(0) is True
    assert fs.should_sample(1) is False
    assert fs.should_sample(2) is True

def test_progress_tracker():
    pt = ProgressTracker()
    pt.start("job1")
    assert pt.get("job1") == 0.0
    
    pt.update("job1", 50.0)
    assert pt.get("job1") == 50.0
    
    pt.update("job1", 150.0)
    assert pt.get("job1") == 100.0
    
    pt.fail("job1")
    assert pt.get("job1") == -1.0

def test_observability_trace():
    tid = start_trace("test_trace")
    assert tid is not None
    assert trace_id_ctx.get() == tid
    
    start_span("test_span")
    # should not crash
    end_span("success")
    
    end_trace("success")
    assert trace_id_ctx.get() is None

def test_observe_decorator():
    @observe("test_func")
    def my_func(x):
        return x + 1
        
    assert my_func(1) == 2
    
    @observe("async_test_func")
    async def my_async_func(x):
        return x + 1
        
    assert asyncio.run(my_async_func(1)) == 2
