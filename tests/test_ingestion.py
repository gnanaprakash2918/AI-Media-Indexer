import pytest
import os
from pathlib import Path
from unittest.mock import patch, MagicMock, AsyncMock

from core.ingestion.pipeline import IngestionPipeline
from core.ingestion.scanner import LibraryScanner
from core.schemas import MediaType

# --- Ingestion Pipeline Tests ---

@pytest.fixture
def mock_pipeline_deps():
    """Mocks heavy dependencies of IngestionPipeline using patch."""
    with patch("core.ingestion.pipeline.MediaProber"), \
         patch("core.ingestion.pipeline.AudioTranscriber"), \
         patch("core.ingestion.pipeline.VisionAnalyzer"), \
         patch("core.ingestion.pipeline.FaceManager"), \
         patch("core.ingestion.pipeline.VoiceProcessor"), \
         patch("core.ingestion.pipeline.VectorDB"), \
         patch("core.ingestion.pipeline.FrameExtractor"):
        yield

@pytest.mark.asyncio
async def test_pipeline_process_video_flow(mock_pipeline_deps, tmp_path):
    video = tmp_path / "test.mp4"
    video.touch()

    pipeline = IngestionPipeline()
    pipeline.prober.probe.return_value = {"format": {"duration": "10.0"}}
    pipeline._process_audio = AsyncMock()
    pipeline._process_voice = AsyncMock()
    pipeline._process_frames = AsyncMock()

    job_id = await pipeline.process_video(video)

    assert job_id is not None
    pipeline.prober.probe.assert_called()
    pipeline._process_audio.assert_called()
    pipeline._process_voice.assert_called()
    pipeline._process_frames.assert_called()

@pytest.mark.asyncio
async def test_pipeline_missing_file(mock_pipeline_deps):
    pipeline = IngestionPipeline()
    with pytest.raises(FileNotFoundError):
        await pipeline.process_video("missing.mp4")

@pytest.mark.asyncio
async def test_process_audio_step(mock_pipeline_deps, tmp_path):
    video = tmp_path / "test.mp4"
    pipeline = IngestionPipeline()
    pipeline.db = MagicMock()

    # Mock AudioTranscriber context manager and methods
    mock_transcriber = MagicMock()
    mock_transcriber.transcribe.return_value = [{"text": "Hello", "start": 0, "end": 1}]

    with patch("core.ingestion.pipeline.AudioTranscriber") as MockTranscriber:
        MockTranscriber.return_value.__enter__.return_value = mock_transcriber

        await pipeline._process_audio(video)

        # Verify db insertion
        pipeline.db.insert_media_segments.assert_called()

@pytest.mark.asyncio
async def test_process_frames_step(mock_pipeline_deps, tmp_path):
    video = tmp_path / "test.mp4"
    pipeline = IngestionPipeline()

    # Mock extract generator
    async def mock_extract(*args, **kwargs):
        f = tmp_path / "frame.jpg"
        f.touch()
        yield f

    pipeline.extractor.extract = mock_extract
    pipeline.frame_sampler = MagicMock()
    pipeline.frame_sampler.should_sample.return_value = True

    # Mock Vision/Faces being instantiated inside _process_frames
    with patch("core.ingestion.pipeline.VisionAnalyzer") as MockVision, \
         patch("core.ingestion.pipeline.FaceManager") as MockFaces:

        mock_vision = AsyncMock()
        mock_vision.describe.return_value = "A scene"
        MockVision.return_value = mock_vision

        mock_faces = MagicMock()
        mock_faces.detect_faces = AsyncMock(return_value=[])
        MockFaces.return_value = mock_faces

        await pipeline._process_frames(video)

        mock_vision.describe.assert_called()
        mock_faces.detect_faces.assert_called()

# --- Library Scanner Tests ---

@pytest.fixture
def scanner():
    return LibraryScanner()

def test_scanner_detects_files(scanner, tmp_path):
    # Setup filesystem
    (tmp_path / "vid.mp4").touch()
    (tmp_path / "song.mp3").touch()
    (tmp_path / "pic.jpg").touch()
    (tmp_path / "ignore.txt").touch()

    ignored_dir = tmp_path / "node_modules"
    ignored_dir.mkdir()
    (ignored_dir / "bad.mp4").touch()

    assets = list(scanner.scan(tmp_path))

    # Check counts
    assert len(assets) == 3

    # Verify Types
    types = {a.media_type for a in assets}
    assert MediaType.VIDEO in types
    assert MediaType.AUDIO in types
    assert MediaType.IMAGE in types

    paths = {a.file_path.name for a in assets}
    assert "vid.mp4" in paths
    assert "bad.mp4" not in paths # Should be ignored

def test_scanner_excludes(scanner, tmp_path):
    (tmp_path / "custom_ignore").mkdir()
    (tmp_path / "custom_ignore" / "vid.mp4").touch()

    assets = list(scanner.scan(tmp_path, excluded_dirs=["custom_ignore"]))
    assert len(assets) == 0

def test_scanner_empty_path(scanner):
    # Empty path should yield no results (logged internally)
    assets = list(scanner.scan(""))
    assert assets == []

def test_scanner_invalid_path(scanner):
    # Non-existent path should yield no results (logged internally)
    assets = list(scanner.scan("non_existent_path_123"))
    assert assets == []
