import pytest
import os
import shutil
from pathlib import Path

# Configuration

def pytest_addoption(parser):
    parser.addoption(
        "--force-clean",
        action="store_true",
        default=False,
        help="Force cleanup without prompts",
    )
    parser.addoption(
        "--llm-provider",
        action="store",
        default="mock",
        help="Specify LLM provider: mock, ollama, gemini",
    )

@pytest.fixture(scope="session")
def force_clean(request):
    return request.config.getoption("--force-clean")

@pytest.fixture
def llm_provider(request):
    return request.config.getoption("--llm-provider")

# Test Data & Media Fixtures

@pytest.fixture(scope="session")
def test_data_dir(tmp_path_factory):
    """
    Creates a temporary directory for test media files.
    """
    return tmp_path_factory.mktemp("media")

@pytest.fixture(scope="function")
def sample_media(test_data_dir):
    """
    Copies distinct sample media (video and audio) to the test directory.
    Uses absolute paths specified by the user.
    """
    # Define source paths
    sources = {
        "video": Path(r"C:\Users\Gnana Prakash M\Downloads\Programs\keladi.webm"),
        "audio": Path(r"C:\Users\Gnana Prakash M\Downloads\Programs\Javed Ali - Siragugal.flac")
    }
    
    results = {}
    
    for key, path in sources.items():
        if not path.exists():
            # Create dummy if missing to avoid crash
            dummy = test_data_dir / f"dummy_{key}{path.suffix}"
            dummy.touch()
            results[key] = dummy
        else:
            dest = test_data_dir / path.name
            shutil.copy2(path, dest)
            results[key] = dest
            
    return results

# Mocks & Environment

@pytest.fixture(autouse=True)
def mock_config(monkeypatch, llm_provider):
    """
    Sets up the environment for testing. 
    Auto-used to ensure NO test runs against prod config.
    """
    # Enforce Test Mode
    monkeypatch.setenv("AI_INDEXER_TEST_MODE", "1")
    monkeypatch.setenv("ENV", "test")
    monkeypatch.setenv("LLM_PROVIDER", llm_provider)
    
    # Mock Database Creds to avoid accidental production writes
    monkeypatch.setenv("QDRANT_HOST", "localhost")
    monkeypatch.setenv("QDRANT_PORT", "6334") # Assuming defaults
    monkeypatch.setenv("POSTGRES_DB", "test_db")
    monkeypatch.setenv("POSTGRES_USER", "test_user")
    monkeypatch.setenv("POSTGRES_PASSWORD", "test_pass")
    
    # Mock AWS/LLM keys to avoid real calls unless integration test requests them
    monkeypatch.setenv("AWS_ACCESS_KEY_ID", "test_aws_key")
    monkeypatch.setenv("AWS_SECRET_ACCESS_KEY", "test_aws_secret")
    monkeypatch.setenv("OPENAI_API_KEY", "test_openai_key")
    
    # Disable heavy logging
    monkeypatch.setenv("LOG_LEVEL", "ERROR")
