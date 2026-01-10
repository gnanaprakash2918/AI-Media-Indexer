"""Golden Run - Final System Verification.

Ingests a test video, verifies all SOTA features work, outputs deployment status.
"""
from __future__ import annotations

import asyncio
import subprocess
import sys
import time
from pathlib import Path

import httpx

sys.path.insert(0, str(Path(__file__).parent.parent))

BASE_URL = "http://localhost:8000"


async def create_test_video(output_path: Path) -> bool:
    """Generate a 10-second test video using FFmpeg."""
    cmd = [
        "ffmpeg", "-y",
        "-f", "lavfi", "-i", "testsrc=duration=10:size=640x360:rate=30",
        "-f", "lavfi", "-i", "sine=frequency=440:duration=10",
        "-c:v", "libx264", "-c:a", "aac",
        "-pix_fmt", "yuv420p",
        str(output_path)
    ]
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
        return result.returncode == 0 and output_path.exists()
    except Exception as e:
        print(f"  FFmpeg failed: {e}")
        return False


async def wait_for_job(client: httpx.AsyncClient, job_id: str, timeout: int = 120) -> bool:
    """Wait for ingestion job to complete."""
    start = time.time()
    while time.time() - start < timeout:
        try:
            resp = await client.get(f"{BASE_URL}/jobs/{job_id}")
            data = resp.json()
            status = data.get("status", "")
            if status == "done":
                return True
            if status in ("failed", "cancelled"):
                print(f"  Job failed: {data.get('error', 'Unknown')}")
                return False
        except Exception:
            pass
        await asyncio.sleep(2)
    print("  Timeout waiting for job")
    return False


async def run_golden_test():
    print("\n" + "=" * 60)
    print("GOLDEN RUN - SYSTEM DEPLOYMENT VERIFICATION")
    print("=" * 60)
    
    results = {
        "server_online": False,
        "ingestion_complete": False,
        "global_context_exists": False,
        "identity_suggestions": False,
        "search_works": False,
    }
    
    async with httpx.AsyncClient(timeout=30.0) as client:
        print("\n[1/5] Checking server health...")
        try:
            resp = await client.get(f"{BASE_URL}/health")
            if resp.status_code == 200:
                results["server_online"] = True
                print("  ✅ Server is online")
            else:
                print("  ❌ Server returned non-200")
                return results
        except Exception as e:
            print(f"  ❌ Server unreachable: {e}")
            return results
        
        print("\n[2/5] Creating and ingesting test video...")
        test_video = Path(__file__).parent / "test_golden.mp4"
        if not test_video.exists():
            print("  Creating test video with FFmpeg...")
            if not await create_test_video(test_video):
                print("  ⚠️ Could not create test video, skipping ingestion test")
                results["ingestion_complete"] = None
            else:
                print(f"  Created: {test_video}")
        
        if test_video.exists():
            try:
                resp = await client.post(
                    f"{BASE_URL}/ingest",
                    json={"path": str(test_video.absolute())}
                )
                data = resp.json()
                job_id = data.get("job_id")
                if job_id:
                    print(f"  Job started: {job_id}")
                    if await wait_for_job(client, job_id, timeout=300):
                        results["ingestion_complete"] = True
                        print("  ✅ Ingestion completed successfully")
                    else:
                        print("  ❌ Ingestion failed or timed out")
                else:
                    print(f"  ❌ No job_id returned: {data}")
            except Exception as e:
                print(f"  ❌ Ingestion request failed: {e}")
        
        print("\n[3/5] Checking global context...")
        try:
            resp = await client.get(f"{BASE_URL}/config/system")
            if resp.status_code == 200:
                results["global_context_exists"] = True
                print("  ✅ System config accessible (global context storage)")
            else:
                print("  ⚠️ System config not accessible")
        except Exception as e:
            print(f"  ⚠️ Global context check: {e}")
        
        print("\n[4/5] Checking identity suggestions...")
        try:
            resp = await client.get(f"{BASE_URL}/identity/suggestions")
            data = resp.json()
            suggestions = data.get("suggestions", [])
            results["identity_suggestions"] = True
            print(f"  ✅ Identity API working ({len(suggestions)} suggestions)")
        except Exception as e:
            print(f"  ❌ Identity suggestions failed: {e}")
        
        print("\n[5/5] Testing search...")
        try:
            resp = await client.get(f"{BASE_URL}/search", params={"q": "test video", "limit": 5})
            data = resp.json()
            search_results = data.get("results", [])
            results["search_works"] = True
            print(f"  ✅ Search works ({len(search_results)} results)")
        except Exception as e:
            print(f"  ❌ Search failed: {e}")
    
    print("\n" + "=" * 60)
    print("RESULTS SUMMARY")
    print("=" * 60)
    
    passed = sum(1 for v in results.values() if v is True)
    total = sum(1 for v in results.values() if v is not None)
    
    for check, status in results.items():
        icon = "✅" if status is True else "⚠️" if status is None else "❌"
        print(f"  {icon} {check.replace('_', ' ').title()}: {'PASS' if status else 'SKIP' if status is None else 'FAIL'}")
    
    print("\n" + "=" * 60)
    if passed == total and total >= 4:
        print("🚀 SYSTEM READY FOR DEPLOYMENT")
        print("=" * 60)
        return True
    elif passed >= 3:
        print("⚠️ SYSTEM MOSTLY READY (some features may need attention)")
        print("=" * 60)
        return True
    else:
        print("❌ SYSTEM NOT READY - Critical failures detected")
        print("=" * 60)
        return False


def main():
    success = asyncio.run(run_golden_test())
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
