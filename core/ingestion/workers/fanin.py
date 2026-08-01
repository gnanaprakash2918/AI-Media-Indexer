"""Redis fan-in counter helpers.

Fan-in pattern:
  - At dispatch time: SET ingest:fanin:expected:{chunk_id} = N
  - Each track on completion (or quarantine): INCR ingest:fanin:{chunk_id}
  - When INCR result == N: enqueue FusionAgent

All operations are atomic Redis commands — no race conditions even with
multiple workers completing simultaneously.

Key TTL is 48 hours (172800 s). Fusion always fires within minutes of
dispatch; 48h gives ample buffer for very slow tracks or paused queues.
"""

from __future__ import annotations

from config import settings
from core.utils.logger import logger

_FANIN_TTL = 172_800  # 48 hours in seconds
_FANIN_KEY = "ingest:fanin:{chunk_id}"
_EXPECTED_KEY = "ingest:fanin:expected:{chunk_id}"


def _redis():
    """Return a synchronous Redis client (used inside Celery sync tasks)."""
    import redis as _redis_lib

    return _redis_lib.Redis(
        host=settings.redis_host,
        port=settings.redis_port,
        password=settings.redis_auth,
        decode_responses=True,
    )


def set_fanin_expected(chunk_id: str, expected_count: int) -> None:
    """Set the expected track count for a chunk. Call once at dispatch time.

    Args:
        chunk_id: Stable chunk identity hash.
        expected_count: Total number of tracks that must complete before
            FusionAgent fires. Excludes disabled optional tracks.
    """
    r = _redis()
    key = _EXPECTED_KEY.format(chunk_id=chunk_id)
    counter_key = _FANIN_KEY.format(chunk_id=chunk_id)
    r.set(key, expected_count, ex=_FANIN_TTL)
    r.set(counter_key, 0, ex=_FANIN_TTL)
    logger.debug(
        f"[FanIn] set expected={expected_count} for chunk={chunk_id[:8]}"
    )


def increment_and_check(chunk_id: str) -> bool:
    """Atomically increment the completion counter and return True if Fusion should fire.

    This is the ONLY place the fan-in counter is incremented. Both
    successful completions AND quarantine transitions call this — so
    FusionAgent is never deadlocked waiting for a dead track.

    Args:
        chunk_id: Stable chunk identity hash.

    Returns:
        True if the counter has reached the expected count (Fusion should fire).
        False otherwise.
    """
    r = _redis()
    counter_key = _FANIN_KEY.format(chunk_id=chunk_id)
    expected_key = _EXPECTED_KEY.format(chunk_id=chunk_id)

    count = r.incr(counter_key)  # atomic: returns new value after increment
    expected_raw = r.get(expected_key)

    if expected_raw is None:
        logger.warning(
            f"[FanIn] No expected count found for chunk={chunk_id[:8]}. "
            "Fan-in key may have expired. Skipping Fusion trigger."
        )
        return False

    expected = int(expected_raw)
    logger.debug(
        f"[FanIn] chunk={chunk_id[:8]} count={count}/{expected}"
    )
    return count == expected


def get_status(chunk_id: str) -> dict:
    """Return current fan-in status for a chunk (debugging only)."""
    r = _redis()
    counter = r.get(_FANIN_KEY.format(chunk_id=chunk_id))
    expected = r.get(_EXPECTED_KEY.format(chunk_id=chunk_id))
    return {
        "chunk_id": chunk_id,
        "completed": int(counter) if counter else 0,
        "expected": int(expected) if expected else None,
    }
