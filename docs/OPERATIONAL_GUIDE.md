# Operational Guide: AI Media Indexer

Production deployment and 18-hour video ingestion guide.

---

## The 18-Hour Flight Checklist

### Pre-Flight Checks

```powershell
# 1. Verify GPU is available
nvidia-smi

# 2. Check VRAM (need 6GB+ free for full pipeline)
python -c "import torch; print(f'VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f}GB')"

# 3. Verify infrastructure
python tests/verify_infrastructure.py

# 4. Check memory stress test passes
python tests/stress_test_memory.py
```

### Docker Launch (Recommended for 18h+ Videos)

```bash
# Start all services
docker-compose up -d

# Verify services are healthy
docker-compose ps

# Check logs
docker-compose logs -f media-indexer
```

### Local Launch (Development)

```powershell
# Terminal 1: Start Qdrant
docker run -p 6333:6333 -p 6334:6334 qdrant/qdrant

# Terminal 2: Start Redis (for Celery)
docker run -p 6379:6379 redis:alpine redis-server --requirepass redispass

# Terminal 3: Start API server
python -m api.server

# Terminal 4: Start Celery worker (for long jobs)
celery -A core.ingestion.celery_app worker --loglevel=info
```

### Start Ingestion

```powershell
# Via API
curl -X POST http://localhost:8000/ingest \
  -H "Content-Type: application/json" \
  -d '{"path": "/path/to/18hour_video.mp4"}'

# Via CLI
python -m core.ingestion.pipeline --path "/path/to/18hour_video.mp4"
```

---

## Monitoring Cockpit

### GPU Monitoring

```powershell
# Real-time VRAM usage (every 2 seconds)
nvidia-smi -l 2

# Or use watch (Linux/WSL)
watch -n 2 nvidia-smi
```

### Celery Worker Logs

```powershell
# Follow Celery logs
celery -A core.ingestion.celery_app inspect active

# Check worker status
celery -A core.ingestion.celery_app status
```

### API Server Logs

```powershell
# Logs are in logs/app.log
tail -f logs/app.log
```

### Docker Logs

```bash
# Follow all logs
docker-compose logs -f

# Just the worker
docker-compose logs -f media-indexer
```

---

## Troubleshooting

### Worker Restarts (Expected Behavior)

**Symptom**: Celery worker restarts after ~50 tasks.

**Cause**: This is **intentional**. `worker_max_tasks_per_child=50` triggers restart to clear memory fragmentation.

**Action**: None needed. Progress is saved. Worker resumes from checkpoint.

### Qdrant Connection Refused (WinError 10053)

**Symptom**: "An established connection was aborted by the software in your host machine."

**Cause**: Windows-specific transient socket closure during heavy I/O.

**Status**: **Wired with Resilience**. The `VectorDB` uses `retry_on_connection_error` (audited in `core/storage/db.py`) to automatically handle these transient failures with exponential backoff.

**Action**: If it persists, ensure `BATCH_SIZE` is reduced to 4 and check for firewall interference.

### 24-Hour Timeout

**Symptom**: Task killed after 24 hours.

**Cause**: `task_time_limit=86400` hard limit.

**Fix**: For videos >24h, split manually:
```powershell
ffmpeg -i input.mp4 -t 43200 -c copy part1.mp4
ffmpeg -i input.mp4 -ss 43200 -c copy part2.mp4
```

### Ingestion Stuck

**Symptom**: No progress for >10 minutes.

**Check**:
1. GPU temperature: `nvidia-smi` (should be <85°C)
2. Disk space: `df -h`
3. Worker alive: `celery -A core.ingestion.celery_app status`

---

## Hardware Profiles

| Profile | VRAM | Batch Size | Estimated Time (18h video) |
|---------|------|------------|---------------------------|
| LAPTOP | <8GB | 4 | ~12 hours |
| WORKSTATION | 8-20GB | 8 | ~6 hours |
| SERVER | >20GB | 16 | ~3 hours |

Auto-detected on startup. Check logs for `Detected [PROFILE] profile`.

---

## Health Endpoints

```
GET /health          # Basic health check
GET /stats           # Ingestion statistics
GET /status/{job_id} # Job status
```
