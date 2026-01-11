
"""Golden Run - Operational Verification Suite.

Verifies the integration of Antigravity features:
1. Hybrid ASR (Native/Docker Switching)
2. VideoRAG (Search Response Structure)
3. Agent System (Connectivity)
"""
import unittest
import sys
import os
from pathlib import Path
from unittest.mock import MagicMock, patch

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.processing.indic_transcriber import IndicASRPipeline
from config import settings

class TestOperationalIntegration(unittest.TestCase):
    
    def test_01_hybrid_asr_switching(self):
        """Verify ASR backend switches based on configuration."""
        print("\nTesting Hybrid ASR Switch logic...")
        
        # Mock dependencies to avoid real loading
        with patch("core.processing.indic_transcriber.IndicASRPipeline.load_model") as mock_load:
             with patch("core.processing.indic_transcriber.HAS_NEMO", True):
                # Case 1: Native Enabled
                settings.use_native_nemo = True
                settings.ai4bharat_url = ""
                
                pipeline = IndicASRPipeline(lang="ta")
                # Trigger transcribe logic verify it sets backend (simulated)
                # We can't easily call transcribe with file input in unit test without file.
                # But we can check internal logic if we refactored it nicely.
                # Given current logic is in `transcribe()`, we test that.
                
                # Create dummy file
                dummy = Path("test.wav")
                dummy.touch()
                try:
                    pipeline.transcribe(dummy)
                    # Implementation details check:
                    # Logic sets `self._backend = "nemo"` if Native
                    self.assertEqual(pipeline._backend, "nemo", "Should prefer Native NeMo")
                finally:
                    dummy.unlink()

    def test_02_videorag_response_structure(self):
        """Verify VideoRAG returns expected fields (match_reasons)."""
        print("Testing VideoRAG structure...")
        
        # We Mock the DB search
        mock_db = MagicMock()
        mock_db.search_frames_hybrid.return_value = [
            {
                "id": "test_id",
                "score": 0.9,
                "video_path": "vid.mp4",
                "match_reasons": ["semantic", "face_match"],
                "entities": ["Prakash"],
                "timestamp": 10.0
            }
        ]
        
        from core.retrieval.rag import VideoRAGOrchestrator, SearchResultItem
        
        orchestrator = VideoRAGOrchestrator(db=mock_db)
        
        # Run search (Async requires sync wrapper or IsolatedAsyncioTestCase)
        import asyncio
        results = asyncio.run(orchestrator._search_multimodal(
             structured=MagicMock(identities=[], visual_cues=["bowling"], audio_cues=[], scene_description="test"),
             limit=1,
             video_path=None
        ))
        
        self.assertTrue(len(results) > 0)
        item = results[0]
        self.assertIsInstance(item, SearchResultItem)
        self.assertIn("semantic", item.match_reasons)
        print("✅ VideoRAG Mock Search passed.")

    def test_03_agent_connectivity(self):
        """Verify Agent Client can init (Phase 12 Fix)."""
        print("Testing Agent connection...")
        from core.agent.client import McpClient
        try:
            client = McpClient()
            # If __init__ fails (e.g. path issues), verification failed.
            self.assertIsNotNone(client)
            print("✅ Agent Client initialized.")
        except Exception as e:
            self.fail(f"Agent Client init failed: {e}")

if __name__ == "__main__":
    unittest.main()
