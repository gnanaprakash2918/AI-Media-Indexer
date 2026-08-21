"""Voice diarization and identity stage.

Extracted from IngestionPipeline to reduce God class size.
IngestionPipeline inherits from VoiceStageMixin to compose these methods.
"""

from __future__ import annotations

import hashlib
import uuid
from pathlib import Path
from typing import TYPE_CHECKING, Any

from config import settings
from core.processing.voice import VoiceProcessor
from core.storage.db import VectorDB
from core.utils.logger import logger
from core.utils.hardware import RESOURCE_ARBITER

if TYPE_CHECKING:
    pass


class VoiceStage:
    """Voice diarization and identity stage."""

    def __init__(self, db: VectorDB, cleanup_memory):
        self.db = db
        self._cleanup_memory = cleanup_memory

    async def process_voice(self, path: Path) -> None:
        """Processes voice diarization and identity registries.

        Extracts voice segments, generates embeddings, matches them against
        the global speaker registry, and stores them in the database. Also
        extracts audio clips for each identified voice segment.

        Args:
            path: Path to the media file.
        """
        voice = VoiceProcessor()

        try:
            voice_segments = await voice.process(path)

            # Prepare voice thumbnails directory
            thumb_dir = settings.cache_dir / "thumbnails" / "voices"
            thumb_dir.mkdir(parents=True, exist_ok=True)
            import subprocess

            # Create safe prefix
            safe_stem = hashlib.md5(path.stem.encode()).hexdigest()

            # Global Speaker Registry Logic
            # 1. Match against existing speakers
            # 2. Assign Global ID
            # 3. Persist specific samples for future matching

            # Cluster IDs now use db.get_next_voice_cluster_id() for uniqueness

            # === P0.4 FIX: GHOST SPEAKER EXPLOSION ===
            # Group segments by local speaker label first
            from collections import defaultdict

            from core.processing.voice import compute_speaker_centroid

            local_speaker_segments = defaultdict(list)
            for seg in voice_segments or []:
                local_speaker_segments[seg.speaker_label].append(seg)

            # Resolve global identity for each local speaker
            local_to_global_map = {}
            local_to_cluster_map = {}

            for local_label, segments in local_speaker_segments.items():
                if local_label == "SILENCE":
                    local_to_global_map[local_label] = "SILENCE"
                    local_to_cluster_map[local_label] = -1
                    continue

                # Compute centroid for this local speaker
                valid_embeddings = [
                    s.embedding for s in segments if s.embedding is not None
                ]
                centroid = compute_speaker_centroid(valid_embeddings)

                global_id = f"unknown_{uuid.uuid4().hex[:8]}"
                cluster_id = -1

                if centroid is not None:
                    # Match CENTROID against global DB (much more stable than single segment)
                    match = await self.db.match_speaker(
                        centroid,
                        threshold=settings.voice_clustering_threshold,
                    )
                    if match:
                        global_id, cluster_id, _ = match
                        if cluster_id == -1:
                            cluster_id = self.db.get_next_voice_cluster_id()
                    else:
                        # New Global Speaker
                        global_id = f"SPK_{uuid.uuid4().hex[:12]}"
                        cluster_id = self.db.get_next_voice_cluster_id()

                        # Register this new speaker with the CENTROID (or first good embedding)
                        # We use the centroid as the representative vector
                        self.db.upsert_speaker_embedding(
                            speaker_id=global_id,
                            embedding=centroid,  # Use centroid as reference
                            media_path=str(path),  # Representative path
                            start=segments[0].start_time,
                            end=segments[0].end_time,
                            voice_cluster_id=cluster_id,
                        )
                        # Also upsert the centroid specifically if we have a collection for it
                        self.db.upsert_voice_cluster_centroid(
                            cluster_id, centroid
                        )

                local_to_global_map[local_label] = global_id
                local_to_cluster_map[local_label] = cluster_id

            for _idx, seg in enumerate(voice_segments or []):
                audio_path: str | None = None

                # Apply resolved global identity
                global_speaker_id = local_to_global_map.get(
                    seg.speaker_label, "unknown"
                )
                voice_cluster_id = local_to_cluster_map.get(
                    seg.speaker_label, -1
                )

                # Store individual segment embedding linked to the cluster
                if seg.embedding is not None and global_speaker_id != "SILENCE":
                    self.db.upsert_speaker_embedding(
                        speaker_id=global_speaker_id,
                        embedding=seg.embedding,
                        media_path=str(path),
                        start=seg.start_time,
                        end=seg.end_time,
                        voice_cluster_id=voice_cluster_id,
                    )

                # ALWAYS extract audio clip for every segment (not just those with embeddings)
                audio_extraction_success = False
                try:
                    clip_name = f"{safe_stem}_{seg.start_time:.2f}_{seg.end_time:.2f}.mp3"
                    clip_file = thumb_dir / clip_name

                    if not clip_file.exists():
                        cmd = [
                            "ffmpeg",
                            "-y",
                            "-i",
                            str(path),
                            "-ss",
                            str(seg.start_time),
                            "-t",
                            str(seg.end_time - seg.start_time),
                            "-q:a",
                            "2",  # High quality MP3
                            "-map",
                            "a",
                            "-loglevel",
                            "error",
                            str(clip_file),
                        ]
                        result = subprocess.run(
                            cmd, capture_output=True, text=True
                        )
                        if result.returncode != 0:
                            logger.warning(
                                f"[Voice] FFmpeg failed ({result.returncode}): {result.stderr[:100]}"
                            )
                except Exception as e:
                    logger.warning(
                        f"[Voice] FFmpeg failed for {clip_name}: {e}"
                    )

                if clip_file.exists() and clip_file.stat().st_size > 0:
                    audio_path = f"/thumbnails/voices/{clip_name}"
                    audio_extraction_success = True
                else:
                    audio_extraction_success = False

                # === STORE VOICE SEGMENT ===
                # Policy: Strict storage - MUST have audio and not be SILENCE
                if not audio_extraction_success:
                    continue

                if global_speaker_id == "SILENCE":
                    continue

                # Speech Emotion Recognition (SER)
                # Gated by settings.enable_speech_emotion (default: False).
                # When disabled: true no-op — no import, no model load, no GPU.
                # To enable: set ENABLE_SPEECH_EMOTION=true in .env and
                #             uv sync --group enrichment to install Wav2Vec2 deps.
                emotion_meta = {}
                if settings.enable_speech_emotion:
                    try:
                        if not hasattr(self, "_ser_analyzer"):
                            self._ser_failed = False
                            try:
                                from core.processing.speech_emotion import (
                                    SpeechEmotionAnalyzer,
                                )

                                self._ser_analyzer = SpeechEmotionAnalyzer()
                            except Exception as init_err:
                                logger.warning(
                                    f"[Voice] SER init failed (disabling): {init_err}"
                                )
                                self._ser_analyzer = None
                                self._ser_failed = True

                        if self._ser_analyzer is not None:
                            import librosa

                            # Load the clip we just made (resample to 16k for Wav2Vec2)
                            y, sr = librosa.load(str(clip_file), sr=16000)
                            emotion_res = await self._ser_analyzer.analyze(y, sr)
                            if emotion_res:
                                emotion_meta = {
                                    "emotion": emotion_res.get("emotion"),
                                    "emotion_conf": emotion_res.get("confidence"),
                                }
                    except Exception as e:
                        if not getattr(self, "_ser_failed", False):
                            logger.warning(f"[Voice] SER failed: {e}")
                            self._ser_failed = True

                if seg.embedding is not None and audio_extraction_success:
                    # Ensure voice_cluster_id is always valid (never -1 or 0)
                    if voice_cluster_id <= 0:
                        voice_cluster_id = self.db.get_next_voice_cluster_id()
                        logger.info(
                            f"[Voice] Generated fallback cluster ID {voice_cluster_id} "
                            f"for segment {seg.start_time:.2f}-{seg.end_time:.2f}s"
                        )

                    self.db.insert_voice_segment(
                        media_path=str(path),
                        start=seg.start_time,
                        end=seg.end_time,
                        speaker_label=global_speaker_id,
                        embedding=seg.embedding.tolist()
                        if hasattr(seg.embedding, "tolist")
                        else seg.embedding,
                        audio_path=audio_path,
                        voice_cluster_id=voice_cluster_id,
                        **emotion_meta,
                    )
                elif seg.embedding is None and audio_extraction_success:
                    # Has audio but no embedding - still useful for playback
                    # Store with placeholder embedding
                    logger.info(
                        f"[Voice] Segment {seg.start_time:.2f}-{seg.end_time:.2f}s "
                        f"has audio but no embedding, storing with placeholder"
                    )
                    # Generate a placeholder cluster ID
                    if voice_cluster_id <= 0:
                        voice_cluster_id = self.db.get_next_voice_cluster_id()

                    # Create a zero embedding placeholder (using small epsilon for safe Cosine distance)
                    placeholder_embedding = [
                        1e-6
                    ] * 256  # WeSpeaker embedding size
                    self.db.insert_voice_segment(
                        media_path=str(path),
                        start=seg.start_time,
                        end=seg.end_time,
                        speaker_label=global_speaker_id,
                        embedding=placeholder_embedding,
                        audio_path=audio_path,
                        voice_cluster_id=voice_cluster_id,
                    )
                else:
                    # No audio AND no embedding - skip completely
                    logger.warning(
                        f"[Voice] Skipping segment {seg.start_time:.2f}-{seg.end_time:.2f}s: "
                        f"no audio (success={audio_extraction_success}), "
                        f"no embedding (has_emb={seg.embedding is not None})"
                    )
        finally:
            voice.cleanup()
            self._cleanup_memory()
