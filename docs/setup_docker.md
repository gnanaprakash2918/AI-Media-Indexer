````md
# AI Media Indexer — Docker + Langfuse Profiles Guide

This project uses **Docker Compose Profiles** to cleanly control whether the **Langfuse self-hosted stack** runs locally, uses **Langfuse Cloud**, or is **disabled entirely**, without changing compose files.

It also standardizes:
- Python dependencies via `requirements.docker.txt`
- Docker build context via `.dockerignore`
- All runtime caches via a project-local `.cache/` directory

---

## Project Structure

```text
.
├── docker-compose.yml
├── .env
├── .dockerignore
├── requirements.docker.txt
├── .cache/                 # ALL runtime caches (gitignored)
├── qdrant_data/
├── postgres_data/
├── clickhouse_data/
├── minio_data/
└── src/
````

---

## Docker Compose Profiles (Core Concept)

Docker Compose **Profiles** allow grouping services and enabling them conditionally.

In this setup:

* **Qdrant** → always runs
* **Langfuse + dependencies** → only run when profile `docker` is active

The active profile is driven by **one environment variable**.

---

## `.env` Configuration (Master Switch)

```env
# --- Langfuse Mode ---
LANGFUSE_BACKEND=docker
# LANGFUSE_BACKEND=cloud
# LANGFUSE_BACKEND=disabled

# --- Docker Profile Bridge ---
COMPOSE_PROFILES=${LANGFUSE_BACKEND}

# --- Local Langfuse ---
LANGFUSE_DOCKER_HOST=http://localhost:3300

# --- Langfuse Cloud (optional) ---
# LANGFUSE_PUBLIC_KEY=pk_...
# LANGFUSE_SECRET_KEY=sk_...
# LANGFUSE_HOST=https://cloud.langfuse.com
```

### How it Works

| LANGFUSE_BACKEND | Active Docker Profile     | Result                   |
| ---------------- | ------------------------- | ------------------------ |
| `docker`         | `docker`                  | Full Langfuse stack runs |
| `cloud`          | `cloud` (none defined)    | Only Qdrant runs         |
| `disabled`       | `disabled` (none defined) | Only Qdrant runs         |

---

## Docker Compose Design

### Always-On Service

```yaml
qdrant:
  image: qdrant/qdrant:v1.9.3
  ports:
    - "6333:6333"
    - "6334:6334"
```

Runs in **all modes**.

---

### Langfuse Stack (Profile: `docker`)

All of the following services are tagged:

```yaml
profiles: ["docker"]
```

* `postgres`
* `clickhouse`
* `redis`
* `minio`
* `createbuckets`
* `langfuse`

They are **ignored** unless `COMPOSE_PROFILES=docker`.

---

## Langfuse (Docker) Internals

### Databases

| Component  | Purpose                   |
| ---------- | ------------------------- |
| PostgreSQL | Metadata, users, projects |
| ClickHouse | Traces, spans, events     |
| Redis      | Queueing                  |
| MinIO      | Event blob storage        |

### Required ClickHouse Configuration

```yaml
CLICKHOUSE_URL: clickhouse://default:langfuse@clickhouse:9000
CLICKHOUSE_MIGRATION_URL: clickhouse://default:langfuse@clickhouse:9000
```

Langfuse v3 **requires native ClickHouse protocol** for migrations.

---

## Python Dependencies (Docker)

### `requirements.docker.txt`

* Used **only** for Docker images
* Keeps host/dev Python clean
* Ensures deterministic builds

Example usage in Dockerfile:

```dockerfile
RUN pip install --no-cache-dir -r requirements.docker.txt
```

---

## Cache Strategy (`.cache/`)

All runtime caches are centralized in:

```text
.cache/
```

Used for:

* Python `__pycache__`
* Torch / model caches
* Temporary runtime files

### Benefits

* Easy cleanup
* Git-safe
* Docker-friendly
* No global cache pollution

Ensure `.cache/` exists locally and is gitignored.

---

## `.dockerignore` (Required)

Ensure at minimum:

```dockerignore
.cache
__pycache__
.git
.gitignore
.env
*.log
qdrant_data
postgres_data
clickhouse_data
minio_data
```

This keeps images:

* Smaller
* Faster to build
* Free of local state

---

## How to Run

### Self-Hosted Langfuse

```env
LANGFUSE_BACKEND=docker
```

```bash
docker compose up -d
```

Result:

* Qdrant runs
* Full Langfuse stack runs
* UI: [http://localhost:3300](http://localhost:3300)

---

### Langfuse Cloud

```env
LANGFUSE_BACKEND=cloud
```

```bash
docker compose up -d --remove-orphans
```

Result:

* Only Qdrant runs
* No local Langfuse infra

---

### Langfuse Disabled

```env
LANGFUSE_BACKEND=disabled
```

```bash
docker compose up -d --remove-orphans
```

Result:

* Only Qdrant runs
* Observability fully off

---

## Useful Commands

```bash
docker compose up -d
docker compose down
docker compose down -v
docker compose ps
docker compose config
docker logs -f media_agent_langfuse
```

---

## Key Guarantees of This Setup

* One compose file
* One environment switch
* No duplicated infra
* No commented YAML
* Cloud/local parity
* Clean cache management
* Production-aligned Langfuse v3 stack

---

## Summary

> **`LANGFUSE_BACKEND` controls everything.**
> Docker Compose Profiles handle the rest.

This is the cleanest, safest way to manage optional infrastructure in Docker.
