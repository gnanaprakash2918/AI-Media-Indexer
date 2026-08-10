import pytest
import torch
import numpy as np
from unittest.mock import AsyncMock, MagicMock, patch

from core.processing.audio_events import extract_feature_tensor


def test_extract_feature_tensor():
    """Test extract_feature_tensor handles various HuggingFace output formats."""
    # 1. Bare Tensor
    tensor = torch.zeros((1, 512))
    assert extract_feature_tensor(tensor) is tensor

    # 2. Object with text_embeds
    class MockTextOutput:
        def __init__(self, t):
            self.text_embeds = t
    assert extract_feature_tensor(MockTextOutput(tensor)) is tensor

    # 3. Object with pooler_output
    class MockPoolerOutput:
        def __init__(self, t):
            self.pooler_output = t
    assert extract_feature_tensor(MockPoolerOutput(tensor)) is tensor

    # 4. Tuple/List
    assert extract_feature_tensor([tensor]) is tensor
    assert extract_feature_tensor((tensor,)) is tensor


@pytest.mark.asyncio
async def test_search_scenes_hybrid_dimension_mismatch():
    """Test that search_scenes skips querying visual_features if dimensions mismatch and encoding fails/is unavailable."""
    from core.storage.repositories.scene_repository import SceneRepository
    
    # Mock qdrant client
    mock_client = MagicMock()
    mock_client.query_points.return_value = MagicMock(points=[])
    
    repo = SceneRepository()
    repo.client = mock_client
    repo.SCENES_COLLECTION = "scenes"
    
    # Mock encode_texts to return a 1024-dim BGE-M3 text vector
    repo.encode_texts = AsyncMock(return_value=[np.zeros(1024).tolist()])
    
    # Patch get_default_visual_encoder to fail or return a vector of wrong size
    with patch("core.storage.repositories.scene_repository.getattr") as mock_getattr:
        # Mock settings.visual_features_dim to 1152 and other settings to True
        def mock_settings_getattr(obj, name, default):
            if name == "visual_features_dim":
                return 1152
            return default
        mock_getattr.side_effect = mock_settings_getattr

        # Execute search_scenes in hybrid mode with a string query
        results = await repo.search_scenes(
            query="test query", 
            search_mode="hybrid", 
            limit=5
        )

        # Verify that query_points was NOT called with using="visual_features"
        for call in mock_client.query_points.call_args_list:
            kwargs = call.kwargs
            assert kwargs.get("using") != "visual_features", "Qdrant was queried with mismatched visual_features dimension!"
            
        assert len(results) == 0


@pytest.mark.asyncio
async def test_search_scenes_hybrid_dimension_success():
    """Test that search_scenes queries visual_features if text query can be encoded to match dimension."""
    from core.storage.repositories.scene_repository import SceneRepository
    
    mock_client = MagicMock()
    mock_client.query_points.return_value = MagicMock(points=[])
    
    repo = SceneRepository()
    repo.client = mock_client
    repo.SCENES_COLLECTION = "scenes"
    repo.encode_texts = AsyncMock(return_value=[np.zeros(1024).tolist()])
    
    # Patch get_default_visual_encoder to succeed with a 1152-dim vector
    mock_encoder = AsyncMock()
    mock_encoder.encode_text.return_value = np.zeros(1152).tolist()
    
    with patch("core.storage.repositories.scene_repository.getattr") as mock_getattr, \
         patch("core.processing.visual_encoder.get_default_visual_encoder", return_value=mock_encoder):
        def mock_settings_getattr(obj, name, default):
            if name == "visual_features_dim":
                return 1152
            return default
        mock_getattr.side_effect = mock_settings_getattr

        # Execute search_scenes in hybrid mode
        await repo.search_scenes(
            query="test query", 
            search_mode="hybrid", 
            limit=5
        )

        # Verify that query_points WAS called with using="visual_features" because dimension matched!
        visual_features_queried = any(
            call.kwargs.get("using") == "visual_features" 
            for call in mock_client.query_points.call_args_list
        )
        assert visual_features_queried is True, "Qdrant was NOT queried for visual_features despite dimension matching!"
