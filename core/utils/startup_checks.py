"""Pre-flight startup checks for the AI Media Indexer.

Verifies all critical dependencies are available before the pipeline
starts. Fails fast with clear error messages instead of cryptic
crashes mid-processing.
"""

from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

from core.utils.logger import get_logger

log = get_logger(__name__)


class StartupCheckError(RuntimeError):
    """Raised when a critical startup check fails."""


def _check_ffmpeg() -> tuple[bool, str]:
    """Verify FFmpeg is installed and accessible."""
    ffmpeg_path = shutil.which("ffmpeg")
    if not ffmpeg_path:
        return (
            False,
            "FFmpeg not found in PATH. Install from https://ffmpeg.org/",
        )
    try:
        result = subprocess.run(
            ["ffmpeg", "-version"],
            capture_output=True,
            text=True,
            timeout=10,
        )
        version_line = (
            result.stdout.split("\n")[0] if result.stdout else "unknown"
        )
        return True, version_line
    except Exception as e:
        return False, f"FFmpeg check failed: {e}"


def _check_ffprobe() -> tuple[bool, str]:
    """Verify FFprobe is installed and accessible."""
    ffprobe_path = shutil.which("ffprobe")
    if not ffprobe_path:
        return False, "FFprobe not found in PATH. Usually bundled with FFmpeg."
    try:
        result = subprocess.run(
            ["ffprobe", "-version"],
            capture_output=True,
            text=True,
            timeout=10,
        )
        version_line = (
            result.stdout.split("\n")[0] if result.stdout else "unknown"
        )
        return True, version_line
    except Exception as e:
        return False, f"FFprobe check failed: {e}"


def _check_qdrant() -> tuple[bool, str]:
    """Verify Qdrant is reachable."""
    from config import settings

    host = settings.qdrant_host
    port = settings.qdrant_port

    try:
        import urllib.request

        url = f"http://{host}:{port}/collections"
        req = urllib.request.Request(url, method="GET")
        with urllib.request.urlopen(req, timeout=5) as resp:
            if resp.status == 200:
                return True, f"http://{host}:{port}"
    except Exception as e:
        return False, (
            f"Qdrant not reachable at http://{host}:{port}. "
            f"Start it with: docker compose -f docker-compose.lite.yaml up qdrant -d\n"
            f"  Error: {e}"
        )
    return False, f"Qdrant returned unexpected response at http://{host}:{port}"


def _check_cuda() -> tuple[bool, str]:
    """Check CUDA availability and VRAM (informational, not critical)."""
    try:
        import torch

        if torch.cuda.is_available():
            name = torch.cuda.get_device_name(0)
            vram = torch.cuda.get_device_properties(0).total_memory / (1024**3)
            return True, f"{name} ({vram:.1f} GB VRAM)"
        return False, "CUDA not available — will use CPU (much slower)"
    except ImportError:
        return False, "PyTorch not installed"


def _check_ollama() -> tuple[bool, str]:
    """Check if Ollama is reachable (optional, for VLM captioning)."""
    from config import settings

    base_url = settings.ollama_base_url

    try:
        import urllib.request

        url = f"{base_url}/api/tags"
        req = urllib.request.Request(url, method="GET")
        with urllib.request.urlopen(req, timeout=5) as resp:
            if resp.status == 200:
                import json

                data = json.loads(resp.read().decode())
                models = [m.get("name", "") for m in data.get("models", [])]
                return (
                    True,
                    f"{len(models)} models available: {', '.join(models[:5])}",
                )
    except Exception:
        return False, (
            f"Ollama not reachable at {base_url}. "
            "VLM captioning will fail. Start Ollama or use Gemini provider."
        )
    return False, f"Ollama returned unexpected response at {base_url}"


def _check_model_cache() -> tuple[bool, str]:
    """Check if the model cache directory exists and has content."""
    from config import settings

    cache_dir = Path(settings.model_cache_dir)
    if not cache_dir.exists():
        cache_dir.mkdir(parents=True, exist_ok=True)
        return True, f"Created: {cache_dir}"

    contents = list(cache_dir.iterdir())
    total_size = sum(
        f.stat().st_size for f in cache_dir.rglob("*") if f.is_file()
    )
    size_gb = total_size / (1024**3)
    return True, f"{len(contents)} items, {size_gb:.2f} GB at {cache_dir}"


def run_startup_checks(*, require_qdrant: bool = True) -> None:
    """Run all startup checks and print a status table.

    Args:
        require_qdrant: If True, fail if Qdrant is not reachable.
            Set to False for offline/test scenarios.

    Raises:
        StartupCheckError: If any critical check fails.
    """
    checks = [
        ("FFmpeg", _check_ffmpeg, True),
        ("FFprobe", _check_ffprobe, True),
        ("Qdrant", _check_qdrant, require_qdrant),
        ("CUDA/GPU", _check_cuda, False),
        ("Ollama", _check_ollama, False),
        ("Model Cache", _check_model_cache, False),
    ]

    results: list[tuple[str, bool, str, bool]] = []
    for name, fn, critical in checks:
        try:
            ok, detail = fn()
        except Exception as e:
            ok, detail = False, f"Check crashed: {e}"
        results.append((name, ok, detail, critical))

    # Print status table
    lines = [
        "",
        "=" * 60,
        f"{'STARTUP STATUS CHECK':^60}",
        "=" * 60,
        f"  {'Component':<15} | {'Status':<8} | {'Detail'}",
        "-" * 60,
    ]
    for name, ok, detail, critical in results:
        status = "OK" if ok else ("FAIL" if critical else "WARN")
        # Truncate detail for display
        detail_short = detail[:40] if len(detail) <= 40 else detail[:37] + "..."
        lines.append(f"  {name:<15} | {status:<8} | {detail_short}")
    lines.append("=" * 60)

    log.info("\n".join(lines))

    # Check for critical failures
    failures = [
        (name, detail)
        for name, ok, detail, critical in results
        if not ok and critical
    ]

    if failures:
        msg_parts = ["\nCRITICAL STARTUP FAILURES:"]
        for name, detail in failures:
            msg_parts.append(f"  ✗ {name}: {detail}")
        msg_parts.append("\nFix these issues before running the pipeline.")
        full_msg = "\n".join(msg_parts)
        log.error(full_msg)
        raise StartupCheckError(full_msg)

    log.info("All critical startup checks passed.")
