import pytest
from core.ingestion.pipeline import IngestionPipeline
from core.storage.db import VectorDB
from core.processing.vision import VisionAnalyzer
from core.processing.transcriber import AudioTranscriber
from config import settings

@pytest.mark.smoke
class TestSmoke:
    """Basic smoke tests to verify key components can be instantiated."""

    def test_config_loads(self):
        assert settings.project_root is not None
        assert settings.cache_dir is not None

    def test_vector_db_init_memory(self):
        db = VectorDB(backend="memory", path=":memory:")
        assert db is not None
        # Should verify collections exist
        collections = [c.name for c in db.client.get_collections().collections]
        assert VectorDB.MEDIA_SEGMENTS_COLLECTION in collections

    def test_pipeline_init(self):
        pipeline = IngestionPipeline(qdrant_backend="memory", qdrant_path=":memory:")
        assert pipeline is not None
        assert pipeline.db is not None
        assert pipeline.prober is not None

    def test_transcriber_init(self):
        # Validate initialization without full model load overhead
        transcriber = AudioTranscriber()
        assert transcriber is not None
        transcriber.unload_model()

    def test_vision_init_mock(self):
        # Vision analyzer tries to load prompt file
        # Assumes prompts/vision_prompt.txt exists
        try:
            vision = VisionAnalyzer()
            assert vision is not None
        except Exception as e:
            pytest.fail(f"VisionAnalyzer init failed: {e}")
