"""Infrastructure Verification Test.

Verifies that all dependencies and configurations are correct for production deployment.

Usage:
    python tests/verify_infrastructure.py
"""

from __future__ import annotations

import sys


def test_library_imports() -> dict:
    """Test that required libraries are importable."""
    print("\n📦 Testing Library Imports...")

    results = {}

    # Audio processing
    try:
        import soundfile

        results["soundfile"] = "✅ OK"
        print(f"   soundfile: ✅ v{soundfile.__version__}")
    except ImportError as e:
        results["soundfile"] = f"❌ {e}"
        print("   soundfile: ❌ Missing (apt-get install libsndfile1)")

    # Computer vision
    try:
        import cv2

        results["cv2"] = "✅ OK"
        print(f"   cv2: ✅ v{cv2.__version__}")
    except ImportError as e:
        results["cv2"] = f"❌ {e}"
        print("   cv2: ❌ Missing (apt-get install libgl1)")

    # PyTorch
    try:
        import torch

        gpu_status = "CUDA" if torch.cuda.is_available() else "CPU only"
        results["torch"] = f"✅ {gpu_status}"
        print(f"   torch: ✅ v{torch.__version__} ({gpu_status})")
    except ImportError as e:
        results["torch"] = f"❌ {e}"
        print("   torch: ❌ Missing")

    # Transformers
    try:
        import transformers

        results["transformers"] = "✅ OK"
        print(f"   transformers: ✅ v{transformers.__version__}")
    except ImportError as e:
        results["transformers"] = f"❌ {e}"
        print("   transformers: ❌ Missing")

    return results


def test_celery_config() -> dict:
    """Test Celery configuration values."""
    print("\n⏱️  Testing Celery Configuration...")

    results = {}

    try:
        from core.ingestion.celery_app import celery_app

        conf = celery_app.conf

        # Check timeout settings
        time_limit = conf.get("task_time_limit")
        soft_limit = conf.get("task_soft_time_limit")
        max_tasks = conf.get("worker_max_tasks_per_child")

        # Verify values
        if time_limit == 86400:
            results["task_time_limit"] = "✅ 24h"
            print(f"   task_time_limit: ✅ {time_limit}s (24 hours)")
        else:
            results["task_time_limit"] = f"⚠️ {time_limit}"
            print(f"   task_time_limit: ⚠️ {time_limit}s (expected 86400)")

        if soft_limit == 82800:
            results["task_soft_time_limit"] = "✅ 23h"
            print(f"   task_soft_time_limit: ✅ {soft_limit}s (23 hours)")
        else:
            results["task_soft_time_limit"] = f"⚠️ {soft_limit}"
            print(f"   task_soft_time_limit: ⚠️ {soft_limit}s (expected 82800)")

        if max_tasks == 50:
            results["worker_max_tasks_per_child"] = "✅ 50"
            print(f"   worker_max_tasks_per_child: ✅ {max_tasks}")
        else:
            results["worker_max_tasks_per_child"] = f"⚠️ {max_tasks}"
            print(f"   worker_max_tasks_per_child: ⚠️ {max_tasks} (expected 50)")

    except Exception as e:
        results["error"] = str(e)
        print(f"   ❌ Failed to load Celery config: {e}")

    return results


def test_hardware_profile() -> dict:
    """Test hardware profile detection."""
    print("\n🖥️  Testing Hardware Profile Detection...")

    results = {}

    try:
        from config import HardwareProfile, get_hardware_profile

        profile = get_hardware_profile()

        profile_name = profile.get("name", "unknown")
        batch_size = profile.get("batch_size", 0)
        device = profile.get("device", "unknown")

        results["profile"] = str(profile_name)
        results["batch_size"] = batch_size
        results["device"] = device

        print(f"   Profile: {profile_name}")
        print(f"   Batch Size: {batch_size}")
        print(f"   Device: {device}")
        print(f"   Worker Count: {profile.get('worker_count', 0)}")

        # Verify profile makes sense
        if profile_name == HardwareProfile.LAPTOP and batch_size == 4:
            print("   ✅ LAPTOP profile correctly configured")
        elif profile_name == HardwareProfile.WORKSTATION and batch_size == 8:
            print("   ✅ WORKSTATION profile correctly configured")
        elif profile_name == HardwareProfile.SERVER and batch_size == 16:
            print("   ✅ SERVER profile correctly configured")
        elif profile_name == HardwareProfile.CPU_ONLY:
            print("   ✅ CPU_ONLY profile (no GPU detected)")

    except Exception as e:
        results["error"] = str(e)
        print(f"   ❌ Failed to detect hardware profile: {e}")

    return results


def main():
    """Run all infrastructure tests."""
    print("=" * 60)
    print("🔧 INFRASTRUCTURE VERIFICATION TEST")
    print("=" * 60)

    all_results = {}

    # Run tests
    all_results["libraries"] = test_library_imports()
    all_results["celery"] = test_celery_config()
    all_results["hardware"] = test_hardware_profile()

    # Summary
    print("\n" + "=" * 60)
    print("📊 SUMMARY")
    print("=" * 60)

    has_errors = False

    # Check for failures
    for category, results in all_results.items():
        for key, value in results.items():
            if isinstance(value, str) and "❌" in value:
                has_errors = True
                print(f"   ❌ {category}.{key}: {value}")

    if not has_errors:
        print("\n✅ All infrastructure checks passed!")
        sys.exit(0)
    else:
        print("\n⚠️  Some checks failed. Review above for details.")
        sys.exit(1)


if __name__ == "__main__":
    main()
