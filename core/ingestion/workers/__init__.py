"""Ingestion worker package.

Workers are thin Celery task wrappers around the existing stage Mixins.
Each worker:
  1. Marks the chunk as PROCESSING in chunk_state.
  2. Calls the underlying stage logic.
  3. Marks COMPLETED and INCRs the Redis fan-in counter.
  4. On failure: marks FAILED; if attempt_count >= max_attempts, marks
     QUARANTINED and INCRs fan-in anyway (so Fusion is never blocked).

Import order matters: celery_app must be imported before tasks so the
Celery application is configured before task decorators are evaluated.
"""
