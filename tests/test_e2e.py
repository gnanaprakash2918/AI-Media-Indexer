import pytest
from pathlib import Path
from core.ingestion.pipeline import IngestionPipeline
from core.retrieval.search import SearchEngine

@pytest.mark.e2e
@pytest.mark.asyncio
async def test_full_pipeline_flow(sample_media, tmp_path, mock_config):
    """
    Runs the full ingestion pipeline on a sample video file.
    Uses 'memory' Qdrant backend for speed.
    """
    video_path = sample_media["video"]
    if not video_path.exists():
        pytest.skip("Sample video not found")

    # Initialize Pipeline
    pipeline = IngestionPipeline(qdrant_backend="memory")

    # Run Processing
    job_id = await pipeline.process_video(video_path, media_type_hint="movie")
    assert job_id is not None

    # Check Database search
    results = pipeline.db.search_media("test")
    assert isinstance(results, list)

@pytest.mark.e2e
@pytest.mark.asyncio
async def test_custom_media_sanity(tmp_path):
    """Sanity check using user-provided paths if they exist."""
    AUDIO_PATH = r"C:\Users\Gnana Prakash M\Downloads\Programs\Javed Ali - Siragugal.flac"
    VIDEO_PATH = r"C:\Users\Gnana Prakash M\Downloads\Programs\Video Song ｜ Keladi Kannmani ｜ S P B ｜ Radhika ｜ Ilaiyaraaja Love Songs [033Z2WNg2Q.webm"
    
    pipeline = IngestionPipeline(qdrant_backend="memory")
    
    # Just check if we can process one of them if present
    for path_str in [AUDIO_PATH, VIDEO_PATH]:
        path = Path(path_str)
        if path.exists():
            job_id = await pipeline.process_video(path)
            assert job_id is not None
            break
    else:
        pytest.skip("User provided media files not found for sanity check")
