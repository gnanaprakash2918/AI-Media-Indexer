import asyncio

# Global semaphore to protect GPU-intensive operations (Pyannote, Whisper)
# to prevent VRAM OOM during parallel ingestion.
GPU_SEMAPHORE = asyncio.Semaphore(1)
