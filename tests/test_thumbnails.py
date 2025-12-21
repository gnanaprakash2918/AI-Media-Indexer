
import shutil
import pytest
from pathlib import Path
from fastapi.testclient import TestClient
from api.server import app
from config import settings

@pytest.fixture
def client():
    return TestClient(app)

@pytest.fixture
def thumbnail_setup():
    # Setup: Create a dummy thumbnail file
    thumb_dir = settings.cache_dir / "thumbnails"
    thumb_dir.mkdir(parents=True, exist_ok=True)
    
    dummy_image = thumb_dir / "test_thumb.jpg"
    dummy_image.write_text("fake image content")
    
    yield "test_thumb.jpg"
    
    # Teardown: Cleanup
    if dummy_image.exists():
        dummy_image.unlink()

def test_thumbnail_static_serving(client, thumbnail_setup):
    """Verify that files in the thumbnail directory are served correctly."""
    filename = thumbnail_setup
    
    # Request the thumbnail
    response = client.get(f"/thumbnails/{filename}")
    
    # Assertions
    assert response.status_code == 200
    assert response.content == b"fake image content"
    # Starlette/FastAPI static files usually guess content type from extension
    assert "image/jpeg" in response.headers["content-type"]

def test_thumbnail_not_found(client):
    """Verify 404 for missing thumbnails."""
    response = client.get("/thumbnails/non_existent.jpg")
    assert response.status_code == 404
