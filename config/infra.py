from typing import Literal
from pydantic import Field


class InfraSettings:
    """Infrastructure, Databases, and Observability settings."""
    
    # Infrastructure (Qdrant)
    qdrant_host: str = Field(default="localhost", description="Qdrant host")
    qdrant_port: int = Field(default=6333, description="Qdrant HTTP port")
    qdrant_backend: str = Field(default="docker", description="'memory' or 'docker'")
    qdrant_timeout: float = Field(
        default=60.0,
        description="Timeout in seconds for Qdrant client operations",
    )
    
    # Redis/Celery Configuration
    redis_host: str = "localhost"
    redis_port: int = 6379
    redis_auth: str = "redispass"
    enable_distributed_ingestion: bool = False
    
    # PostgreSQL — chunk_state idempotent checkpoint table
    postgres_url: str = Field(
        default="postgresql+asyncpg://postgres:postgres@localhost:5432/ai_media",
        description=(
            "Async SQLAlchemy URL for the chunk_state PostgreSQL table. "
            "Alembic migrations strip the +asyncpg prefix automatically."
        ),
    )
    
    # Resource Monitoring
    enable_resource_monitoring: bool = True
    max_cpu_percent: float = 90.0
    max_ram_percent: float = 95.0
    max_temp_celsius: float = 85.0  # Pause if CPU hits 85°C
    max_vram_percent: float = Field(
        default=90.0,
        description="Max GLOBAL VRAM usage before throttling (90% prevents OOM crashes)",
    )
    cool_down_seconds: int = 30
    
    # Langfuse Configuration
    langfuse_backend: Literal["docker", "cloud", "disabled"] = Field(
        default="disabled",
        description="Langfuse backend selection",
    )
    # Cloud Langfuse
    langfuse_public_key: str | None = None
    langfuse_secret_key: str | None = None
    langfuse_host: str = "https://cloud.langfuse.com"
    # Local (Docker) Langfuse
    langfuse_docker_host: str = "http://localhost:3300"
    
    # Observability (Loki)
    enable_loki: bool = Field(default=False, description="Enable log shipping to Grafana Loki")
    loki_url: str = Field(
        default="http://localhost:3100/loki/api/v1/push",
        description="Loki push API URL",
    )
