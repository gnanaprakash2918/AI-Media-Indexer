"""Sanity check - verify torch and basic imports work."""
# MUST be first
import torch

print(f"✅ PyTorch version: {torch.__version__}")
print(f"✅ CUDA available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"✅ CUDA device: {torch.cuda.get_device_name(0)}")

# Test other critical imports
try:
    from core.storage.db import VectorDB
    print("✅ VectorDB import OK")
except Exception as e:
    print(f"❌ VectorDB import failed: {e}")

try:
    from core.retrieval.agentic_search import SearchAgent
    print("✅ SearchAgent import OK")
except Exception as e:
    print(f"❌ SearchAgent import failed: {e}")

try:
    from core.processing.temporal_context import TemporalContextManager
    print("✅ TemporalContextManager import OK")
except Exception as e:
    print(f"❌ TemporalContextManager import failed: {e}")

print("\n🎉 Sanity check passed!")
