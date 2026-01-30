"""Perceptual Hashing for Video Content ID (YouTube Content ID style).

Implements fingerprinting for duplicate/near-duplicate detection.
Robust to rotation, cropping, and speed changes.

Based on Research:
- YouTube Content ID: Audio/Video fingerprinting
- Shazam Visual Search: Feature point hashing
"""

from __future__ import annotations

import asyncio

import numpy as np

from core.utils.logger import get_logger

log = get_logger(__name__)


class PerceptualHasher:
    """Generate perceptual hashes for video frames/audio.

    Uses pHash (perceptual hash) for robust duplicate detection.
    Two similar images will have similar hashes (low Hamming distance).

    Usage:
        hasher = PerceptualHasher()
        hash1 = await hasher.hash_frame(frame1)
        hash2 = await hasher.hash_frame(frame2)
        distance = hasher.hamming_distance(hash1, hash2)
        is_duplicate = distance < 10  # Threshold
    """

    def __init__(self, hash_size: int = 16):
        """Initialize perceptual hasher.

        Args:
            hash_size: Size of the hash (default 16 = 256-bit hash).
        """
        self.hash_size = hash_size
        self._imagehash = None
        self._init_lock = asyncio.Lock()

    async def _lazy_load(self) -> bool:
        """Load imagehash library lazily."""
        if self._imagehash is not None:
            return True

        async with self._init_lock:
            if self._imagehash is not None:
                return True

            try:
                import imagehash

                self._imagehash = imagehash
                log.info("[PerceptualHash] imagehash library loaded")
                return True
            except ImportError:
                log.warning(
                    "[PerceptualHash] imagehash not installed (pip install imagehash)"
                )
                return False

    async def hash_frame(self, frame: np.ndarray) -> str | None:
        """Generate perceptual hash for a frame.

        Args:
            frame: RGB/BGR frame as numpy array.

        Returns:
            Hex string hash, or None if failed.
        """
        if not await self._lazy_load():
            return None

        try:
            from PIL import Image

            # Convert to PIL Image
            if isinstance(frame, np.ndarray):
                image = Image.fromarray(frame)
            else:
                image = frame

            # Generate pHash (perceptual hash)
            phash = self._imagehash.phash(image, hash_size=self.hash_size)
            return str(phash)

        except Exception as e:
            log.error(f"[PerceptualHash] Hash failed: {e}")
            return None

    async def hash_video_keyframes(
        self,
        frames: list[np.ndarray],
        sample_interval: int = 5,
    ) -> list[str]:
        """Generate hashes for video keyframes.

        Args:
            frames: List of video frames.
            sample_interval: Sample every Nth frame.

        Returns:
            List of perceptual hashes.
        """
        hashes = []
        for i, frame in enumerate(frames):
            if i % sample_interval == 0:
                h = await self.hash_frame(frame)
                if h:
                    hashes.append(h)
        return hashes

    def hamming_distance(self, hash1: str, hash2: str) -> int:
        """Calculate Hamming distance between two hashes.

        Lower distance = more similar.
        Distance < 10 typically indicates near-duplicate.

        Args:
            hash1: First hash string.
            hash2: Second hash string.

        Returns:
            Hamming distance (number of differing bits).
        """
        if not hash1 or not hash2:
            return 999  # Max distance for invalid

        try:
            # Convert hex to binary and count differences
            if self._imagehash:
                h1 = self._imagehash.hex_to_hash(hash1)
                h2 = self._imagehash.hex_to_hash(hash2)
                return h1 - h2  # imagehash overloads subtraction for Hamming

            # Fallback: manual calculation
            int1 = int(hash1, 16)
            int2 = int(hash2, 16)
            xor = int1 ^ int2
            return bin(xor).count("1")

        except Exception as e:
            log.error(f"[PerceptualHash] Distance calc failed: {e}")
            return 999

    def is_duplicate(
        self,
        hash1: str,
        hash2: str,
        threshold: int = 10,
    ) -> bool:
        """Check if two hashes indicate duplicate content.

        Args:
            hash1: First hash.
            hash2: Second hash.
            threshold: Maximum Hamming distance for duplicate.

        Returns:
            True if frames are likely duplicates.
        """
        return self.hamming_distance(hash1, hash2) < threshold

    async def find_duplicates(
        self,
        target_hash: str,
        candidate_hashes: list[tuple[str, str]],  # (id, hash)
        threshold: int = 10,
    ) -> list[tuple[str, int]]:
        """Find duplicate frames from a set of candidates.

        Args:
            target_hash: Hash to search for.
            candidate_hashes: List of (id, hash) tuples.
            threshold: Maximum distance threshold.

        Returns:
            List of (id, distance) for matches.
        """
        matches = []
        for cid, chash in candidate_hashes:
            distance = self.hamming_distance(target_hash, chash)
            if distance < threshold:
                matches.append((cid, distance))

        # Sort by distance (closest first)
        matches.sort(key=lambda x: x[1])
        return matches


class AudioFingerprinter:
    """Audio fingerprinting for Content ID (Shazam style).

    Creates robust audio fingerprints for matching songs/sounds
    even with background noise or distortion.
    """

    def __init__(self):
        """Initialize audio fingerprinter."""
        self._chromaprint = None
        self._init_lock = asyncio.Lock()

    async def _lazy_load(self) -> bool:
        """Load chromaprint/acoustid library."""
        if self._chromaprint is not None:
            return True

        async with self._init_lock:
            if self._chromaprint is not None:
                return True

            try:
                import chromaprint

                self._chromaprint = chromaprint
                log.info("[AudioFingerprint] chromaprint loaded")
                return True
            except ImportError:
                log.warning("[AudioFingerprint] chromaprint not installed")
                return False

    async def fingerprint_audio(
        self,
        audio_data: np.ndarray,
        sample_rate: int = 44100,
    ) -> str | None:
        """Generate fingerprint for audio segment.

        Args:
            audio_data: Audio samples as numpy array.
            sample_rate: Sample rate of audio.

        Returns:
            Fingerprint string, or None if failed.
        """
        if not await self._lazy_load():
            return None

        try:
            # Convert to int16 for chromaprint
            if audio_data.dtype != np.int16:
                audio_int16 = (audio_data * 32767).astype(np.int16)
            else:
                audio_int16 = audio_data

            # Generate fingerprint
            fingerprint = self._chromaprint.fingerprint(
                audio_int16.tobytes(),
                sample_rate,
            )
            return fingerprint

        except Exception as e:
            log.error(f"[AudioFingerprint] Failed: {e}")
            return None
