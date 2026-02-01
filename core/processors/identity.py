"""Identity processing module for the media ingestion pipeline."""

from __future__ import annotations

import logging
import subprocess
import uuid
from pathlib import Path

import torch

from config import settings
from core.ingestion.diarization import VoiceProcessor
from core.processing.speech_emotion import SpeechEmotionAnalyzer
from core.utils.observe import observe
from core.utils.resource import resource_manager

logger = logging.getLogger(__name__)


class IdentityProcessor:
    """Handles voice diarization, speaker identification, and identity graph updates."""

    def __init__(self, db):
        self.db = db
        self.voice_processor = None
        self._ser_analyzer = None

    @observe("voice_processing")
    async def process_voice(self, path: Path) -> None:
        """Processes voice diarization and identity registries.

        Extracts voice segments, generates embeddings, matches them against
        the global speaker registry, and stores them in the database.
        """
        await resource_manager.throttle_if_needed("compute")
        self.voice_processor = VoiceProcessor()

        try:
            voice_segments = await self.voice_processor.process(path)

            thumb_dir = settings.cache_dir / "thumbnails" / "voices"
            thumb_dir.mkdir(parents=True, exist_ok=True)

            import hashlib

            safe_stem = hashlib.md5(path.stem.encode()).hexdigest()

            for _idx, seg in enumerate(voice_segments or []):
                audio_path: str | None = None
                global_speaker_id = f"unknown_{uuid.uuid4().hex[:8]}"
                voice_cluster_id = -1

                if seg.speaker_label == "SILENCE":
                    global_speaker_id = "SILENCE"

                if seg.embedding is not None and global_speaker_id != "SILENCE":
                    match = await self.db.match_speaker(
                        seg.embedding,
                        threshold=settings.voice_clustering_threshold,
                    )
                    if match:
                        global_speaker_id, existing_cluster_id, _score = match
                        voice_cluster_id = existing_cluster_id
                        if voice_cluster_id == -1:
                            voice_cluster_id = (
                                self.db.get_next_voice_cluster_id()
                            )
                    else:
                        global_speaker_id = f"SPK_{uuid.uuid4().hex[:12]}"
                        voice_cluster_id = self.db.get_next_voice_cluster_id()

                        self.db.upsert_speaker_embedding(
                            speaker_id=global_speaker_id,
                            embedding=seg.embedding,
                            media_path=str(path),
                            start=seg.start_time,
                            end=seg.end_time,
                            voice_cluster_id=voice_cluster_id,
                        )

                # Audio Extraction
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
                            "2",
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
                                f"[Voice] FFmpeg failed: {result.stderr[:100]}"
                            )

                    if clip_file.exists() and clip_file.stat().st_size > 0:
                        audio_path = f"/thumbnails/voices/{clip_name}"
                        audio_extraction_success = True
                except Exception as e:
                    logger.warning(f"[Voice] Extraction failed: {e}")

                if not audio_extraction_success:
                    continue

                if global_speaker_id == "SILENCE":
                    continue

                # SER Analysis
                emotion_meta = {}
                try:
                    if not self._ser_analyzer:
                        self._ser_analyzer = SpeechEmotionAnalyzer()

                    import librosa

                    if clip_file.exists():
                        y, sr = librosa.load(str(clip_file), sr=16000)
                        emotion_res = await self._ser_analyzer.analyze(y, sr)
                        if emotion_res:
                            emotion_meta = {
                                "emotion": emotion_res.get("emotion"),
                                "emotion_conf": emotion_res.get("confidence"),
                            }
                except Exception as e:
                    logger.warning(f"[Voice] SER failed: {e}")

                # Store Segment
                if seg.embedding is not None:
                    if voice_cluster_id <= 0:
                        voice_cluster_id = self.db.get_next_voice_cluster_id()

                    embedding_list = (
                        seg.embedding.tolist()
                        if hasattr(seg.embedding, "tolist")
                        else seg.embedding
                    )
                    self.db.insert_voice_segment(
                        media_path=str(path),
                        start=seg.start_time,
                        end=seg.end_time,
                        speaker_label=global_speaker_id,
                        embedding=embedding_list,
                        audio_path=audio_path,
                        voice_cluster_id=voice_cluster_id,
                        **emotion_meta,
                    )
                elif audio_extraction_success:
                    # Placeholder embedding
                    if voice_cluster_id <= 0:
                        voice_cluster_id = self.db.get_next_voice_cluster_id()
                    placeholder = [1e-6] * 256
                    self.db.insert_voice_segment(
                        media_path=str(path),
                        start=seg.start_time,
                        end=seg.end_time,
                        speaker_label=global_speaker_id,
                        embedding=placeholder,
                        audio_path=audio_path,
                        voice_cluster_id=voice_cluster_id,
                    )

        finally:
            if self.voice_processor:
                self.voice_processor.cleanup()
            self.voice_processor = None
            import gc

            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
