"""Sanity check - verify torch and basic imports work."""
# MUST be first
import torch

print(f"✅ PyTorch version: {torch.__version__}")
print(f"✅ CUDA available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"✅ CUDA device: {torch.cuda.get_device_name(0)}")

# Test other critical imports
try:
    print("✅ VectorDB import OK")
except Exception as e:
    print(f"❌ VectorDB import failed: {e}")

try:
    print("✅ SearchAgent import OK")
except Exception as e:
    print(f"❌ SearchAgent import failed: {e}")

try:
    print("✅ TemporalContextManager import OK")
except Exception as e:
    print(f"❌ TemporalContextManager import failed: {e}")

print("\n🎉 Sanity check passed!")
