"""Custom Exception Hierarchy for AI-Media-Indexer.

This module defines specific error types to allow for granular error handling
and better debugging. All custom exceptions inherit from `MediaIndexerError`.
"""

class MediaIndexerError(Exception):
    """Base exception for all AI-Media-Indexer errors."""
    def __init__(self, message: str, original_error: Exception | None = None, context: dict | None = None):
        super().__init__(message)
        self.original_error = original_error
        self.context = context or {}

class IngestionError(MediaIndexerError):
    """Raised when the ingestion pipeline fails."""
    pass

class ExtractionError(MediaIndexerError):
    """Raised when frame extraction fails."""
    pass

class TranscriberError(MediaIndexerError):
    """Raised when audio transcription fails."""
    pass

class VisionError(MediaIndexerError):
    """Raised when vision analysis models fail."""
    pass

class ModelLoadError(MediaIndexerError):
    """Raised when a model fails to load."""
    pass

class ResourceError(MediaIndexerError):
    """Raised when system resources (VRAM/RAM/Disk) are exhausted."""
    pass

class DatabaseError(MediaIndexerError):
    """Raised when database operations fail."""
    pass
