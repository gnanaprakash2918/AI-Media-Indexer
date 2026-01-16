import asyncio
import os
import sys

sys.path.insert(0, str(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from loguru import logger
logger.remove()
logger.add(sys.stderr, level="INFO", format="{time:HH:mm:ss} | {level:<7} | {message}")


async def test_overlays_file_exists():
    logger.info("Testing overlays.py file exists and has correct structure...")
    from pathlib import Path
    import re

    file_path = Path(__file__).parent.parent / "api" / "routes" / "overlays.py"
    assert file_path.exists(), f"overlays.py not found at {file_path}"

    content = file_path.read_text(encoding="utf-8")

    assert "get_video_overlays" in content, "Missing get_video_overlays endpoint"
    assert "get_frame_overlays" in content, "Missing get_frame_overlays endpoint"
    assert '"/overlays/{video_id}"' in content, "Missing /overlays/{video_id} route"
    assert '"faces"' in content, "Missing faces overlay type"
    assert '"text_regions"' in content, "Missing text_regions overlay type"
    assert '"objects"' in content, "Missing objects overlay type"

    logger.success("overlays.py structure verified")


async def test_audio_events_open_vocab():
    logger.info("Testing audio_events.py is open-vocabulary...")
    from pathlib import Path
    import re

    file_path = Path(__file__).parent.parent / "core" / "processing" / "audio_events.py"
    content = file_path.read_text(encoding="utf-8")

    hardcoded_patterns = [
        r'SOUND_CLASSES\s*[:=]\s*\[',
        r'"siren",\s*"alarm"',
        r'"dog barking",\s*"cat meowing"',
    ]

    for pattern in hardcoded_patterns:
        match = re.search(pattern, content)
        if match:
            logger.error(f"FOUND HARDCODED LIST: {pattern}")
            raise AssertionError(f"Hardcoded list found: {pattern}")

    assert "target_classes: list[str]" in content, "Missing target_classes parameter"
    assert "match_audio_description" in content, "Missing match_audio_description method"

    logger.success("audio_events.py is OPEN-VOCABULARY - no hardcoded lists")


async def test_overlay_color_scheme():
    logger.info("Testing overlay color scheme...")
    from pathlib import Path

    file_path = Path(__file__).parent.parent / "api" / "routes" / "overlays.py"
    content = file_path.read_text(encoding="utf-8")

    expected_colors = {
        "faces (green)": "#22C55E",
        "text_regions (blue)": "#3B82F6",
        "objects (red)": "#EF4444",
        "active_speakers (yellow)": "#FBBF24",
    }

    for overlay_type, color in expected_colors.items():
        assert color in content, f"Missing color {color} for {overlay_type}"
        logger.info(f"  {overlay_type}: {color} ✓")

    logger.success("Overlay color scheme verified")


async def test_server_includes_overlays():
    logger.info("Testing server.py includes overlays router...")
    from pathlib import Path

    file_path = Path(__file__).parent.parent / "api" / "server.py"
    content = file_path.read_text(encoding="utf-8")

    assert "from api.routes import overlays" in content, "Missing overlays import"
    assert "overlays.router" in content, "Missing overlays router registration"

    logger.success("server.py includes overlays router")


async def test_clothing_no_hardcoded_lists():
    logger.info("Testing clothing_attributes.py has no hardcoded lists...")
    from pathlib import Path
    import re

    file_path = Path(__file__).parent.parent / "core" / "processing" / "clothing_attributes.py"
    content = file_path.read_text(encoding="utf-8")

    hardcoded_patterns = [
        r'CLOTHING_COLORS\s*=\s*\[',
        r'CLOTHING_ITEMS\s*=\s*\[',
        r'CLOTHING_PATTERNS\s*=\s*\[',
    ]

    for pattern in hardcoded_patterns:
        match = re.search(pattern, content)
        if match:
            raise AssertionError(f"Hardcoded list found: {pattern}")

    assert "match_description" in content, "Missing match_description method"
    assert "target_description: str" in content, "Missing target_description parameter"

    logger.success("clothing_attributes.py is OPEN-VOCABULARY")


async def main():
    logger.info("=" * 60)
    logger.info("PHASE 3 VERIFICATION: API & Hardcode Sweep")
    logger.info("=" * 60)

    failed = []

    try:
        await test_overlays_file_exists()
    except Exception as e:
        logger.error(f"OVERLAYS FILE FAILED: {e}")
        failed.append("overlays_file")

    try:
        await test_audio_events_open_vocab()
    except Exception as e:
        logger.error(f"AUDIO EVENTS FAILED: {e}")
        failed.append("audio_events")

    try:
        await test_overlay_color_scheme()
    except Exception as e:
        logger.error(f"OVERLAY COLORS FAILED: {e}")
        failed.append("overlay_colors")

    try:
        await test_server_includes_overlays()
    except Exception as e:
        logger.error(f"SERVER INCLUDES FAILED: {e}")
        failed.append("server_includes")

    try:
        await test_clothing_no_hardcoded_lists()
    except Exception as e:
        logger.error(f"CLOTHING OPEN VOCAB FAILED: {e}")
        failed.append("clothing_open_vocab")

    logger.info("=" * 60)
    if failed:
        logger.error(f"FAILED: {failed}")
        sys.exit(1)
    else:
        logger.success("ALL PHASE 3 API TESTS PASSED")
        sys.exit(0)


if __name__ == "__main__":
    asyncio.run(main())
