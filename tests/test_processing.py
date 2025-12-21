import pytest
import cv2
import numpy as np
import asyncio
from pathlib import Path
from unittest.mock import MagicMock, patch, AsyncMock

from core.processing.extractor import FrameExtractor
from core.processing.identity import FaceManager
from core.processing.metadata import MetadataEngine
from core.processing.prober import MediaProber, MediaProbeError
from core.processing.transcriber import AudioTranscriber
from core.processing.vision import VisionAnalyzer
from core.processing.voice import VoiceProcessor
from core.schemas import MediaType

@pytest.fixture
def mock_ffprobe_deps(monkeypatch):
    monkeypatch.setattr("shutil.which", lambda x: "/usr/bin/ffprobe")

@pytest.mark.asyncio
async def test_prober_basic(mock_ffprobe_deps, tmp_path):
    f = tmp_path / "test.mp4"
    f.touch()
    
    mp = MediaProber()
    with patch("subprocess.Popen") as mock_popen:
        process_mock = MagicMock()
        process_mock.communicate.return_value = ('{"format": {"duration": "10.5"}}', "")
        process_mock.returncode = 0
        mock_popen.return_value = process_mock
        
        meta = mp.probe(f)
        assert float(meta["format"]["duration"]) == 10.5

@pytest.mark.asyncio
async def test_extractor_init():
    ext = FrameExtractor()
    assert ext is not None

@pytest.fixture
def mock_face_deps(monkeypatch):
    monkeypatch.setattr("core.processing.identity.cv2.FaceDetectorYN.create", MagicMock())
    monkeypatch.setattr("core.processing.identity.cv2.FaceRecognizerSF.create", MagicMock())

@pytest.mark.asyncio
async def test_face_manager_cluster(mock_face_deps):
    with patch("core.processing.identity._ensure_models"), \
         patch("core.processing.identity._migrate_cache"):
        fm = FaceManager()
        
        # Mock clustering
        with patch("core.processing.identity.DBSCAN") as mock_dbscan:
            mock_inst = MagicMock()
            mock_inst.fit_predict.return_value = np.array([0, 0, -1])
            mock_dbscan.return_value = mock_inst
            
            embeddings = [np.zeros(128) for _ in range(3)]
            labels = fm.cluster_faces(embeddings)
            assert len(labels) == 3
            assert labels[2] == -1

def test_metadata_identify():
    engine = MetadataEngine(tmdb_api_key="fake")
    # Test filename parsing
    title, year = engine._parse_filename("The.Matrix.1999.1080p.mkv")
    assert title == "The Matrix"
    assert year == 1999

@pytest.mark.asyncio
async def test_vision_describe(tmp_path):
    mock_llm = AsyncMock()
    mock_llm.describe_image.return_value = "a cat"
    mock_llm.construct_user_prompt = MagicMock(return_value="mock prompt")
    
    with patch("core.processing.vision.LLMFactory.create_llm", return_value=mock_llm):
        vis = VisionAnalyzer()
        img = tmp_path / "fake.jpg"
        img.touch()
        desc = await vis.describe(img)
        assert "a cat" in desc

@pytest.mark.asyncio
async def test_transcriber_basic(tmp_path):
    # AudioTranscriber uses Faster Whisper's WhisperModel
    with patch("core.processing.transcriber.WhisperModel"), \
         patch("core.processing.transcriber.BatchedInferencePipeline") as MockPipeline:
        mock_pipeline = MagicMock()
        mock_pipeline.transcribe.return_value = ([], MagicMock())
        MockPipeline.return_value = mock_pipeline
        
        ts = AudioTranscriber()
        audio = tmp_path / "fake.wav"
        audio.touch()
        with patch.object(ts, "_slice_audio", return_value=audio), \
             patch.object(ts, "_convert_and_cache_model", return_value="fake_path"):
            ts.transcribe(audio)
            assert ts._model is not None


@pytest.mark.asyncio
async def test_voice_processor_lifecycle(tmp_path):
    # Mock settings to enable voice
    with patch("config.settings.enable_voice_analysis", True), \
         patch("config.settings.hf_token", "fake_token"):
         
        vp = VoiceProcessor()
        assert vp.enabled is True
        
        # Test lazy init with mocks
        with patch("core.processing.voice.Pipeline.from_pretrained") as mock_pipe, \
             patch("core.processing.voice.Model.from_pretrained") as mock_model, \
             patch("core.processing.voice.Inference") as mock_inf:
             
            mock_pipe.return_value.to.return_value = None
            
            # Create a dummy audio file
            audio = tmp_path / "test.wav"
            audio.touch()
            
            # Mock convert to wav
            with patch.object(vp, "_convert_to_wav", return_value=audio):
                # Mock pipeline processing (diarization)
                mock_diarization = MagicMock()
                # Mock iterator: yields (turn, track, speaker)
                turn = MagicMock()
                turn.start = 0.0
                turn.end = 2.0
                mock_diarization.itertracks.return_value = [(turn, None, "Speaker1")]
                
                # Configure the factory to return our mock pipeline
                mock_pipeline_instance = MagicMock()
                mock_pipeline_instance.return_value = mock_diarization
                
                # When Pipeline.from_pretrained() is called, it returns our pipeline instance
                mock_pipe.return_value = mock_pipeline_instance
                
                vp.inference = MagicMock()
                
                # Mock embedding extraction
                with patch.object(vp, "_extract_embedding", return_value=[0.1]*256):
                     segments = await vp.process(audio)
                     
                     assert len(segments) == 1
                     assert segments[0].speaker_label == "Speaker1"
