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

import hmac
import hashlib
import json
from celery.signals import before_task_publish, task_prerun
from celery.exceptions import Reject

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

HMAC_SECRET = settings.api_key.encode() if getattr(settings, "api_key", None) else b"dev-secret-key-do-not-use-in-prod"

@before_task_publish.connect
def sign_task_payload(sender=None, body=None, **kwargs):
    """Sign the payload before sending to Redis to prevent injection."""
    if body:
        # Body is typically a tuple containing args and kwargs
        payload_str = json.dumps(body, sort_keys=True)
        signature = hmac.new(HMAC_SECRET, payload_str.encode(), hashlib.sha256).hexdigest()
        # Attach the signature to the headers
        headers = kwargs.get('headers') or {}
        headers['X-Task-Signature'] = signature
        kwargs['headers'] = headers

@task_prerun.connect
def verify_task_payload(task_id=None, task=None, args=None, kwargs=None, **kw):
    """Verify the payload signature before executing in the worker."""
    req = task.request
    expected_body = (args, kwargs, req.embed)
    payload_str = json.dumps(expected_body, sort_keys=True)
    expected_sig = hmac.new(HMAC_SECRET, payload_str.encode(), hashlib.sha256).hexdigest()
    
    received_sig = req.headers.get('X-Task-Signature') if req.headers else None
    
    if not received_sig or not hmac.compare_digest(expected_sig, received_sig):
        # We REJECT (and arguably drop) the unsigned or tampered task
        logger = task.logger
        logger.error(f"[SECURITY] Task {task_id} failed HMAC signature verification!")
        raise Reject("Invalid payload signature.", requeue=False)

if __name__ == "__main__":
    celery_app.start()
