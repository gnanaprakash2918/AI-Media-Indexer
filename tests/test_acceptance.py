import pytest
import os
from pathlib import Path
from core.ingestion.pipeline import IngestionPipeline
from core.retrieval.search import SearchEngine
from core.agent.a2a_server import check_ollama_connection

@pytest.mark.acceptance
@pytest.mark.asyncio
async def test_user_story_ingest_and_search(tmp_path, sample_media):
    """
    Story: User drops a video, system processes it, user searches and finds it.
    """
    # 1. Setup
    video_path = sample_media["video"]
    if not video_path.exists() or video_path.name.startswith("dummy"):
        pytest.skip("Real sample video not available for acceptance test")

    # Use memory backend for acceptance to avoid wiping prod/docker DB
    pipeline = IngestionPipeline(qdrant_backend="memory", qdrant_path=":memory:")
    
    # 2. Ingest
    has_ollama = False
    try:
        check_ollama_connection("llama3") # Or default model
        has_ollama = True
    except:
        pass
        
    # Mock dependencies for deterministic output and speed
    
    from unittest.mock import AsyncMock, MagicMock, patch
    
    with patch("core.ingestion.pipeline.VisionAnalyzer") as MockVision, \
         patch("core.ingestion.pipeline.AudioTranscriber") as MockAudio, \
         patch("core.ingestion.pipeline.FaceManager") as MockFace, \
         patch("core.ingestion.pipeline.VoiceProcessor") as MockVoice:
         
        # Configure mock vision to return specific content we can search for
        vision_instance = AsyncMock()
        vision_instance.describe.return_value = "A cute cat playing with a yarn ball."
        MockVision.return_value = vision_instance
        
        # Mock Audio
        audio_instance = MagicMock()
        audio_instance.__enter__ = MagicMock(return_value=audio_instance)
        audio_instance.__exit__ = MagicMock(return_value=None)
        audio_instance.transcribe.return_value = [
            {"text": "Look at the kitty", "start": 0.0, "end": 2.0}
        ]
        MockAudio.return_value = audio_instance
        
        # Mock others
        MockFace.return_value = MagicMock()
        MockFace.return_value.detect_faces = AsyncMock(return_value=[])
        MockVoice.return_value = MagicMock()
        MockVoice.return_value.process = AsyncMock(return_value=[])
        
        # Run Ingestion
        job_id = await pipeline.process_video(video_path)
        assert job_id is not None
        
        # 3. Search
        search_engine = SearchEngine(pipeline.db)
        
        # Search for Visual
        results = search_engine.search("cat")
        assert len(results["visual_matches"]) > 0
        assert "cat" in results["visual_matches"][0]["content"]
        
        # Search for Dialogue
        results = search_engine.search("kitty")
        assert len(results["dialogue_matches"]) > 0
        assert "kitty" in results["dialogue_matches"][0]["content"]

from unittest.mock import MagicMock
