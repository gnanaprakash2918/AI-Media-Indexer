import pytest
from unittest.mock import MagicMock, patch
from core.storage.db import VectorDB
from core.retrieval.search import SearchEngine

@pytest.fixture
def mock_qdrant(monkeypatch):
    """Mocks Qdrant client for isolation."""
    mock_client = MagicMock()
    # Mock collection existence checks
    mock_client.collection_exists.return_value = False

    # Mock search results to return valid structure
    mock_point = MagicMock()
    mock_point.id = "test_id"
    mock_point.score = 0.95
    mock_point.payload = {"text": "match", "start": 0.0, "end": 1.0}

    mock_search_resp = MagicMock()
    mock_search_resp.points = [mock_point]
    mock_client.query_points.return_value = mock_search_resp

    # Mock connection
    monkeypatch.setattr("core.storage.db.QdrantClient", lambda **kwargs: mock_client)
    return mock_client

@pytest.fixture
def mock_encoder(monkeypatch):
    """Mocks SentenceTransformer for speed/isolation."""
    mock_model = MagicMock()
    mock_model.encode.return_value = [[0.1] * 384] # Correct dim for all-MiniLM

    # We mock the internal _load_encoder method to return our mock
    # This avoids downloading models during tests
    monkeypatch.setattr("core.storage.db.VectorDB._load_encoder", lambda self: mock_model)
    return mock_model

def test_db_init_creates_collections(mock_qdrant, mock_encoder):
    db = VectorDB(backend="memory")
    # Verify all expected collections are created
    assert mock_qdrant.create_collection.call_count == 4
    mock_qdrant.get_collections.return_value = [] # Just ensuring call works

def test_insert_media_segments(mock_qdrant, mock_encoder):
    db = VectorDB(backend="memory")
    segments = [{"text": "Hello", "start": 0.0, "end": 2.0}]

    db.insert_media_segments("video.mp4", segments)

    mock_qdrant.upsert.assert_called_once()
    call_args = mock_qdrant.upsert.call_args
    assert call_args.kwargs['collection_name'] == VectorDB.MEDIA_SEGMENTS_COLLECTION
    assert len(call_args.kwargs['points']) == 1

def test_search_media(mock_qdrant, mock_encoder):
    db = VectorDB(backend="memory")
    results = db.search_media("query")

    assert len(results) == 1
    assert results[0]['text'] == "match"
    mock_qdrant.query_points.assert_called()

def test_db_backend_docker(monkeypatch):
    """Verify Docker backend initialization logic."""
    mock_client_cls = MagicMock()
    monkeypatch.setattr("core.storage.db.QdrantClient", mock_client_cls)

    # Needs to mock _load_encoder as well
    monkeypatch.setattr("core.storage.db.VectorDB._load_encoder", lambda s: MagicMock())

    VectorDB(backend="docker", host="localhost", port=6333)
    mock_client_cls.assert_called_with(host="localhost", port=6333)

@pytest.fixture
def mock_db():
    return MagicMock()

def test_search_engine_initialization(mock_db):
    engine = SearchEngine(mock_db)
    assert engine.db == mock_db

def test_search_aggregation(mock_db):
    engine = SearchEngine(mock_db)

    # Mock DB returns generic dicts as per db.py interface (usually list of dicts)
    mock_db.search_frames.return_value = [
        {"timestamp": 10.5, "score": 0.9, "video_path": "vid1.mp4", "action": "running"}
    ]
    mock_db.search_media.return_value = [
        {"start": 5.0, "score": 0.8, "video_path": "vid1.mp4", "text": "hello"}
    ]

    results = engine.search("query", limit=5)

    assert "visual_matches" in results
    assert "dialogue_matches" in results

    # Check visuals
    assert len(results["visual_matches"]) == 1
    v = results["visual_matches"][0]
    assert v["time"] == "10.50s"
    assert v["content"] == "running"

    # Check dialogue
    assert len(results["dialogue_matches"]) == 1
    d = results["dialogue_matches"][0]
    assert d["time"] == "5.00s"
    assert d["content"] == "hello"

def test_search_empty(mock_db):
    engine = SearchEngine(mock_db)
    mock_db.search_frames.return_value = []
    mock_db.search_media.return_value = []

    results = engine.search("query")
    assert results["visual_matches"] == []
    assert results["dialogue_matches"] == []
