"""Celery application configuration for distributed task processing."""

from celery import Celery

from config import settings

# Redis URL from environment or settings
REDIS_URL = f"redis://:{settings.redis_auth}@{settings.redis_host}:{settings.redis_port}/0"

celery_app = Celery(
    "ai_media_indexer",
    broker=REDIS_URL,
    backend=REDIS_URL,
    include=["core.ingestion.tasks"],
)

celery_app.conf.update(
    task_serializer="json",
    accept_content=["json"],
    result_serializer="json",
    timezone="UTC",
    enable_utc=True,
    # Worker settings
    worker_concurrency=settings.max_concurrent_jobs,
    worker_prefetch_multiplier=1,
    task_acks_late=True,
    # Long-form video support (18+ hours)
    task_time_limit=86400,  # 24 hours hard limit
    task_soft_time_limit=82800,  # 23 hours soft limit (graceful cleanup)
    worker_max_tasks_per_child=50,  # Restart worker every 50 tasks (memory fragmentation prevention)
)

if __name__ == "__main__":
    celery_app.start()
