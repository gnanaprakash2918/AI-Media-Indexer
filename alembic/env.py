"""Alembic migration environment.

Reads database URL from application settings so migrations always target
the same database as the running application.

The URL is converted to the synchronous psycopg2 dialect for Alembic itself
(asyncpg is only used by the async application runtime).
"""

from __future__ import annotations

import os
from logging.config import fileConfig

from sqlalchemy import engine_from_config, pool

from alembic import context

# ---------------------------------------------------------------------------
# Alembic Config object
# ---------------------------------------------------------------------------

config = context.config

if config.config_file_name is not None:
    fileConfig(config.config_file_name)

# Import the SQLAlchemy MetaData that defines our tables.
# db_models.py holds plain Table() definitions (no ORM declarative) so Alembic
# can autogenerate migrations without pulling in the full application.
from core.storage.db_models import metadata as target_metadata  # noqa: E402


def _get_url() -> str:
    """Resolve database URL from settings, falling back to env var.

    asyncpg URLs (postgresql+asyncpg://) are converted to psycopg2
    (postgresql://) for Alembic — Alembic requires a synchronous driver.
    """
    try:
        from config import settings  # type: ignore[import]

        url: str = settings.postgres_url
    except Exception:
        url = os.environ.get(
            "DATABASE_URL",
            "postgresql://postgres:postgres@localhost:5432/ai_media",
        )
    return url.replace("postgresql+asyncpg://", "postgresql://")


def run_migrations_offline() -> None:
    """Run migrations in 'offline' mode (emits SQL without a live DB)."""
    url = _get_url()
    context.configure(
        url=url,
        target_metadata=target_metadata,
        literal_binds=True,
        dialect_opts={"paramstyle": "named"},
        compare_type=True,
    )
    with context.begin_transaction():
        context.run_migrations()


def run_migrations_online() -> None:
    """Run migrations in 'online' mode (connects to a live database)."""
    cfg = config.get_section(config.config_ini_section) or {}
    cfg["sqlalchemy.url"] = _get_url()

    connectable = engine_from_config(
        cfg,
        prefix="sqlalchemy.",
        poolclass=pool.NullPool,
    )

    with connectable.connect() as connection:
        context.configure(
            connection=connection,
            target_metadata=target_metadata,
            compare_type=True,
        )
        with context.begin_transaction():
            context.run_migrations()


if context.is_offline_mode():
    run_migrations_offline()
else:
    run_migrations_online()
