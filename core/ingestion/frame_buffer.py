"""Frame buffer for batch database writes during ingestion."""

from __future__ import annotations

from core.utils.logger import logger


class FrameBuffer:
    """Buffers frame data for batch database writes.

    Accumulates processed frame data and flushes to DB in batches
    of `batch_size` for ~10x performance over individual writes.
    """

    def __init__(self, db, batch_size: int | None = None):
        from config import settings

        self.db = db
        self.batch_size = batch_size or settings.embedding_batch_size or 50
        self._buffer: list[dict] = []
        self._total_flushed = 0

    def add(self, frame_data: dict) -> int:
        """Add a frame to the buffer. Returns frames flushed (0 or batch_size).

        Validates timestamp before adding. Skips frames with invalid timestamps.
        """
        timestamp = frame_data.get("timestamp")
        if timestamp is None or (isinstance(timestamp, (int, float)) and timestamp < 0):
            video_path = frame_data.get("video_path", "unknown")
            logger.warning(
                f"[FrameBuffer] Skipping frame with invalid timestamp={timestamp} "
                f"from {video_path}"
            )
            return 0

        self._buffer.append(frame_data)
        if len(self._buffer) >= self.batch_size:
            return self.flush()
        return 0

    def flush(self) -> int:
        """Flush all buffered frames to database."""
        if not self._buffer:
            return 0
        count = self.db.upsert_media_frames_batch(self._buffer)
        self._total_flushed += count
        self._buffer.clear()
        return count

    @property
    def pending(self) -> int:
        """Number of frames waiting to be flushed."""
        return len(self._buffer)

    @property
    def total_written(self) -> int:
        """Total frames actually flushed to database (excludes pending buffer)."""
        return self._total_flushed
