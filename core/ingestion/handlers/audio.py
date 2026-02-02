"""Audio Processing Handler for Ingestion Pipeline."""

from __future__ import annotations

import asyncio
import gc
import re
import traceback
import subprocess
import uuid
from collections import defaultdict
from collections.abc import Iterable
from pathlib import Path
from typing import Any

import numpy as np
import torch
from config import settings
from core.errors import MediaIndexerError
from core.processing.audio_events import get_audio_detector
from core.processing.text_utils import parse_srt
from core.processing.transcriber import AudioTranscriber
from core.processing.voice import VoiceProcessor, compute_speaker_centroid
from core.storage.db import VectorDB
from core.utils.logger import log, logger
from core.utils.progress import progress_tracker
from core.utils.resource import resource_manager
from core.utils.resource_arbiter import RESOURCE_ARBITER
from core.processing.prober import MediaProber

class AudioHandler:
    """Handles audio-related ingestion tasks: Transcription, Voice, and Audio Events."""

    def __init__(self, db: VectorDB, prober: MediaProber):
        self.db = db
        self.prober = prober
        self.voice_processor: VoiceProcessor | None = None
        self.audio_classification: dict[str, float] | None = None
        # Simple local cache for probe data to avoid repeated ffprobe calls within the handler
        self._probe_cache: dict[str, dict] = {}

    async def get_probe_data(self, path: Path) -> dict:
        """Get probe data with caching."""
        key = str(path)
        if key not in self._probe_cache:
            self._probe_cache[key] = await self.prober.probe(path)
        return self._probe_cache[key]

    async def process_audio(self, path: Path) -> dict[str, float] | None:
        """Processes audio to generate transcriptions and language classification.

        Returns:
            Classification metadata (speech/music percentages) if successful.
        """
        audio_segments: list[dict[str, Any]] = []
        srt_path = path.with_suffix(".srt")

        # Check for existing sidecar SRT
        if srt_path.exists():
            audio_segments = parse_srt(srt_path) or []
            if audio_segments:
                log(f"[Audio] Using existing SRT: {len(audio_segments)} segments")

        # Check for embedded subtitles
        if not audio_segments:
            await resource_manager.throttle_if_needed("compute")
            try:
                with AudioTranscriber() as transcriber:
                    temp_srt = path.with_suffix(".embedded.srt")
                    if transcriber._find_existing_subtitles(path, temp_srt, None, "ta"):
                        audio_segments = parse_srt(temp_srt) or []
                        if audio_segments:
                            log(f"[Audio] Extracted embedded subs: {len(audio_segments)} segments")
                        if temp_srt.exists():
                            temp_srt.unlink()
            except Exception as e:
                log(f"[Audio] Embedded subtitle extraction failed: {e}")

        # Run ASR if no existing subtitles
        if not audio_segments:
            await resource_manager.throttle_if_needed("compute")

            # Content Classification (speech/music/silence detection)
            use_lyrics_mode = False
            try:
                from core.processing.content_classifier import get_content_classifier
                classifier = get_content_classifier()
                content_regions = classifier.classify(path)

                if content_regions:
                    use_lyrics_mode = classifier.should_use_lyrics_mode(content_regions)
                    if use_lyrics_mode:
                        log("[Audio] High music content detected - will use lyrics mode")
            except Exception as e:
                log(f"[Audio] Content classification skipped: {e}")

            # Auto-detect language if enabled
            detected_lang = "en"
            detection_confidence = 0.0

            if settings.auto_detect_language:
                detected_lang, detection_confidence = await self.detect_audio_language_with_confidence(path)
                log(f"[Audio] Detected language: {detected_lang} ({detection_confidence:.1%} confidence)")

                # DYNAMIC MULTI-PASS DETECTION
                if detection_confidence < 0.6:
                    log(f"[Audio] Low confidence ({detection_confidence:.1%}), trying second pass...")
                    second_lang, second_conf = await self.detect_audio_language_with_confidence(
                        path, start_offset=30.0, duration=30.0
                    )
                    log(f"[Audio] Second pass: {second_lang} ({second_conf:.1%})")

                    if second_conf > detection_confidence:
                        detected_lang = second_lang
                        detection_confidence = second_conf
                        log(f"[Audio] Using second pass result: {detected_lang}")

                    # Third pass
                    if detection_confidence < 0.5:
                        try:
                            probed = await self.get_probe_data(path)
                            duration = float(probed.get("format", {}).get("duration", 0.0))
                            if duration > 120:
                                mid_point = duration / 2
                                third_lang, third_conf = await self.detect_audio_language_with_confidence(
                                    path, start_offset=mid_point, duration=30.0
                                )
                                log(f"[Audio] Third pass: {third_lang} ({third_conf:.1%})")
                                if third_conf > detection_confidence:
                                    detected_lang = third_lang
                                    detection_confidence = third_conf
                                    log(f"[Audio] Using mid-video detection: {detected_lang}")
                        except Exception:
                            pass
            else:
                detected_lang = settings.language or "en"

            # Choose transcriber based on language
            indic_languages = ["ta", "hi", "te", "ml", "kn", "bn", "gu", "mr", "or", "pa"]

            if settings.use_indic_asr and detected_lang in indic_languages:
                # Use AI4Bharat for Indic languages
                log(f"[Audio] Attempting AI4Bharat IndicConformer for '{detected_lang}'")
                indic_transcriber = None
                try:
                    from core.processing.indic_transcriber import IndicASRPipeline
                    indic_transcriber = IndicASRPipeline(lang=detected_lang)
                    log(f"[Audio] IndicASR backend: {indic_transcriber._backend}")

                    srt_path = path.with_suffix(".srt")
                    audio_segments = await indic_transcriber.transcribe(path, output_srt=srt_path) or []

                    if audio_segments:
                        log(f"[Audio] AI4Bharat SUCCESS: {len(audio_segments)} segments")
                        log(f"[Audio] SRT saved to: {srt_path}")
                    else:
                        raise ValueError("AI4Bharat returned no segments")
                except Exception as e:
                    log(f"[Audio] AI4Bharat failed: {e}")
                    log(f"[Audio] Falling back to Whisper for '{detected_lang}'")
                    try:
                        with AudioTranscriber() as transcriber:
                            async with RESOURCE_ARBITER.acquire("whisper", vram_gb=1.5):
                                audio_segments = await transcriber.transcribe(path, language=detected_lang) or []
                    except Exception as e2:
                        log(f"[Audio] Whisper fallback also failed: {e2}")
                finally:
                    if indic_transcriber is not None:
                        indic_transcriber.unload_model()
            else:
                # Use Whisper for English and other languages
                log(f"[Audio] Using Whisper turbo for '{detected_lang}'" + (" (lyrics mode)" if use_lyrics_mode else ""))
                try:
                    with AudioTranscriber() as transcriber:
                        async with RESOURCE_ARBITER.acquire("whisper", vram_gb=1.5):
                            audio_segments = await transcriber.transcribe(
                                path, language=detected_lang, force_lyrics=use_lyrics_mode
                            ) or []

                        if not audio_segments and not use_lyrics_mode:
                            log("[Audio] No segments found, retrying with lyrics mode...")
                            audio_segments = await transcriber.transcribe(
                                path, language=detected_lang, force_lyrics=True
                            ) or []
                except Exception as e:
                    log(f"[Audio] Whisper failed: {e}")

            if audio_segments:
                log(f"[Audio] Transcription SUCCESS: {len(audio_segments)} segments")
            else:
                log(f"[Audio] WARNING - NO SEGMENTS produced for {path.name}")
                # NEVER-EMPTY GUARANTEE
                try:
                    probed = await self.get_probe_data(path)
                    duration = float(probed.get("format", {}).get("duration", 0.0))
                    if duration > 0:
                        audio_segments = [{"text": "[No speech detected]", "start": 0.0, "end": duration}]
                        log(f"[Audio] Created placeholder segment for {duration:.1f}s media")
                except Exception:
                    pass

        if audio_segments:
            prepared = self._prepare_segments_for_db(path=path, chunks=audio_segments)
            await self.db.insert_media_segments(str(path), prepared)
            log(f"[Audio] Stored {len(prepared)} dialogue segments in DB")

            try:
                probed = await self.get_probe_data(path)
                total_duration = float(probed.get("format", {}).get("duration", 0.0))
                if total_duration > 0:
                    speech_duration = sum(
                        (s.get("end", 0) - s.get("start", 0))
                        for s in audio_segments
                        if s.get("text", "").strip() and "[No speech" not in s.get("text", "")
                    )
                    speech_pct = (speech_duration / total_duration) * 100
                    music_pct = 100 - speech_pct
                    self.audio_classification = {
                        "speech_percentage": min(speech_pct, 100),
                        "music_percentage": max(music_pct, 0),
                        "total_duration": total_duration,
                    }
                    log(f"[Audio] Classification: {speech_pct:.0f}% speech, {music_pct:.0f}% music/ambience")
            except Exception:
                pass

        # === MUSIC STRUCTURE ANALYSIS ===
        await self._process_music_structure(path)

        # === LOUDNESS ANALYSIS ===
        await self._process_loudness(path)

        return self.audio_classification

    async def _process_music_structure(self, path: Path) -> None:
        """Analyze music structure (verse/chorus)."""
        try:
            from core.processing.audio_structure import get_music_analyzer
            log("[MusicStructure] Starting structure analysis...")
            music_analyzer = get_music_analyzer()

            import librosa
            max_duration = 300.0  # 5 minutes max
            
            # Load audio (blocking IO in thread would be better but librosa is pure python/C)
            # Using asyncio.to_thread just in case
            def load_audio():
                return librosa.load(str(path), sr=22050, mono=True, duration=max_duration)

            audio_array, sr = await asyncio.to_thread(load_audio)
            
            log(f"[MusicStructure] Loaded {len(audio_array)/sr:.1f}s audio (limited to {max_duration}s)")
            
            analysis = await asyncio.to_thread(music_analyzer.analyze_array, audio_array, sr=22050)

            if analysis.sections:
                log(f"[MusicStructure] Found {len(analysis.sections)} sections at {analysis.global_tempo:.1f} BPM")
                for section in analysis.sections:
                    self.db.insert_audio_event(
                        media_path=str(path),
                        event_type=f"music_{section.section_type}",
                        start_time=section.start_time,
                        end_time=section.end_time,
                        confidence=section.confidence,
                        payload={
                            "section_type": section.section_type,
                            "energy": section.energy,
                            "beat_count": section.beat_count,
                            "tempo": section.tempo,
                        },
                    )

                self.db.update_media_metadata(
                    media_path=str(path),
                    metadata={
                        "music_tempo": analysis.global_tempo,
                        "has_vocals": analysis.has_vocals,
                        "section_count": len(analysis.sections),
                        "music_structure": [s.to_dict() for s in analysis.sections[:20]],
                    },
                )
                log(f"[MusicStructure] Indexed {len(analysis.sections)} sections")
            else:
                log("[MusicStructure] No sections detected")

        except Exception as e:
            log(f"[MusicStructure] Analysis failed: {e}")
            self._cleanup_memory()

    async def _process_loudness(self, path: Path) -> None:
        """Analyze audio loudness."""
        try:
            log("[Loudness] Starting audio level analysis (FFmpeg streaming)...")
            cmd = [
                "ffmpeg", "-i", str(path), "-af", "ebur128=framelog=verbose:peak=true", "-f", "null", "-"
            ]
            result = await asyncio.to_thread(
                lambda: subprocess.run(cmd, capture_output=True, text=True, timeout=300)
            )

            stderr = result.stderr
            lufs = -23.0
            peak_db = 0.0

            lufs_match = re.search(r"I:\s*(-?\d+\.?\d*)\s*LUFS", stderr)
            if lufs_match:
                lufs = float(lufs_match.group(1))

            peak_match = re.search(r"Peak:\s*(-?\d+\.?\d*)\s*dBFS", stderr)
            if peak_match:
                peak_db = float(peak_match.group(1))

            estimated_spl = max(0, 85 + lufs + 23)
            category = "quiet"
            if estimated_spl >= 85: category = "very_loud"
            elif estimated_spl >= 75: category = "loud"
            elif estimated_spl >= 60: category = "moderate"

            log(f"[Loudness] Overall: {estimated_spl:.0f} dB SPL ({category}) [LUFS: {lufs:.1f}]")

            self.db.update_media_metadata(
                media_path=str(path),
                metadata={
                    "loudness_lufs": lufs,
                    "peak_db": peak_db,
                    "estimated_spl": estimated_spl,
                    "loudness_category": category,
                },
            )
        except Exception as e:
            log(f"[Loudness] Analysis failed: {e}")

    async def process_voice(self, path: Path) -> None:
        """Processes voice diarization and identity registries."""
        await resource_manager.throttle_if_needed("compute")
        self.voice_processor = VoiceProcessor()

        try:
            voice_segments = await self.voice_processor.process(path)
            thumb_dir = settings.cache_dir / "thumbnails" / "voices"
            thumb_dir.mkdir(parents=True, exist_ok=True)
            
            import hashlib
            safe_stem = hashlib.md5(path.stem.encode()).hexdigest()

            # Global Speaker Registry Logic
            local_speaker_segments = defaultdict(list)
            for seg in voice_segments or []:
                local_speaker_segments[seg.speaker_label].append(seg)

            local_to_global_map = {}
            local_to_cluster_map = {}

            for local_label, segments in local_speaker_segments.items():
                if local_label == "SILENCE":
                    local_to_global_map[local_label] = "SILENCE"
                    local_to_cluster_map[local_label] = -1
                    continue

                valid_embeddings = [s.embedding for s in segments if s.embedding is not None]
                centroid = compute_speaker_centroid(valid_embeddings)

                global_id = f"unknown_{uuid.uuid4().hex[:8]}"
                cluster_id = -1

                if centroid is not None:
                    match = await self.db.match_speaker(centroid, threshold=settings.voice_clustering_threshold)
                    if match:
                        global_id, cluster_id, _ = match
                        if cluster_id == -1:
                            cluster_id = self.db.get_next_voice_cluster_id()
                    else:
                        global_id = f"SPK_{uuid.uuid4().hex[:12]}"
                        cluster_id = self.db.get_next_voice_cluster_id()
                        
                        self.db.upsert_speaker_embedding(
                            speaker_id=global_id,
                            embedding=centroid,
                            media_path=str(path),
                            start=segments[0].start_time,
                            end=segments[0].end_time,
                            voice_cluster_id=cluster_id,
                        )
                        self.db.upsert_voice_cluster_centroid(cluster_id, centroid)

                local_to_global_map[local_label] = global_id
                local_to_cluster_map[local_label] = cluster_id

            for _idx, seg in enumerate(voice_segments or []):
                audio_path: str | None = None
                global_speaker_id = local_to_global_map.get(seg.speaker_label, "unknown")
                voice_cluster_id = local_to_cluster_map.get(seg.speaker_label, -1)

                if seg.embedding is not None and global_speaker_id != "SILENCE":
                     self.db.upsert_speaker_embedding(
                        speaker_id=global_speaker_id,
                        embedding=seg.embedding,
                        media_path=str(path),
                        start=seg.start_time,
                        end=seg.end_time,
                        voice_cluster_id=voice_cluster_id,
                    )

                # Audio Clip Extraction
                audio_extraction_success = False
                try:
                    clip_name = f"{safe_stem}_{seg.start_time:.2f}_{seg.end_time:.2f}.mp3"
                    clip_file = thumb_dir / clip_name

                    if not clip_file.exists():
                        cmd = [
                            "ffmpeg", "-y", "-i", str(path), "-ss", str(seg.start_time),
                            "-t", str(seg.end_time - seg.start_time),
                            "-q:a", "2", "-map", "a", "-loglevel", "error", str(clip_file)
                        ]
                        result = subprocess.run(cmd, capture_output=True, text=True)
                        if result.returncode != 0:
                            logger.warning(f"[Voice] FFmpeg failed: {result.stderr[:100]}")
                    
                    if clip_file.exists() and clip_file.stat().st_size > 0:
                        audio_path = f"/thumbnails/voices/{clip_name}"
                        audio_extraction_success = True
                except Exception as e:
                    logger.warning(f"[Voice] FFmpeg failed for {clip_name}: {e}")

                if not audio_extraction_success or global_speaker_id == "SILENCE":
                    continue

                # Speech Emotion Recognition (SER)
                emotion_meta = {}
                try:
                     if not hasattr(self, "_ser_analyzer"):
                         from core.processing.speech_emotion import SpeechEmotionAnalyzer
                         self._ser_analyzer = SpeechEmotionAnalyzer()
                     
                     import librosa
                     y, sr = librosa.load(str(clip_file), sr=16000)
                     emotion_res = await self._ser_analyzer.analyze(y, sr)
                     if emotion_res:
                         emotion_meta = {
                             "emotion": emotion_res.get("emotion"),
                             "emotion_conf": emotion_res.get("confidence"),
                         }
                except Exception as e:
                    logger.warning(f"SER failed: {e}")

                if seg.embedding is not None:
                     if voice_cluster_id <= 0:
                         voice_cluster_id = self.db.get_next_voice_cluster_id()
                     
                     self.db.insert_voice_segment(
                        media_path=str(path),
                        start=seg.start_time,
                        end=seg.end_time,
                        speaker_label=global_speaker_id,
                        embedding=seg.embedding.tolist() if hasattr(seg.embedding, "tolist") else seg.embedding,
                        audio_path=audio_path,
                        voice_cluster_id=voice_cluster_id,
                        **emotion_meta
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
                        **emotion_meta
                    )

        finally:
            if self.voice_processor:
                self.voice_processor.cleanup()
                self.voice_processor = None
            self._cleanup_memory()

    async def process_audio_events(self, path: Path, job_id: str | None = None) -> None:
        """Detects and indexes discrete audio events (CLAP)."""
        logger.info(f"Starting audio event detection for {path.name}")
        try:
            from core.processing.audio_events import get_audio_detector
            detector = get_audio_detector()

            # Duration
            try:
                probe_cmd = ["ffprobe", "-v", "quiet", "-show_entries", "format=duration", "-of", "csv=p=0", str(path)]
                duration = float(subprocess.check_output(probe_cmd).decode().strip())
            except Exception:
                import librosa
                duration = librosa.get_duration(path=str(path))

            chunk_seconds = 30
            overlap_seconds = 5
            stride_seconds = chunk_seconds - overlap_seconds
            sample_rate = 48000
            total_chunks = max(1, int(duration / stride_seconds) + 1)
            previous_events = []
            
            clap_window = 5.0
            clap_stride = 2.5

            for chunk_idx in range(total_chunks):
                chunk_start = chunk_idx * stride_seconds
                chunk_end = min(chunk_start + chunk_seconds, duration)
                if chunk_start >= duration: break

                # Progress Update
                if job_id:
                     progress = 45 + (chunk_idx / total_chunks) * 10
                     progress_tracker.update(
                         job_id, 
                         progress, 
                         stage="audio_events", 
                         message=f"Detecting audio events: chunk {chunk_idx + 1}/{total_chunks}"
                     )

                # Stream chunk
                audio_chunk = await self._extract_audio_segment(path, chunk_start, chunk_end, sample_rate)
                if audio_chunk is None or len(audio_chunk) == 0: continue

                # Split into CLAP windows
                samples_per_window = int(clap_window * sample_rate)
                stride_samples = int(clap_stride * sample_rate)
                clap_chunks = []
                for i in range(0, len(audio_chunk) - samples_per_window + 1, stride_samples):
                    window = audio_chunk[i : i + samples_per_window]
                    window_start = chunk_start + (i / sample_rate)
                    clap_chunks.append((window, window_start))

                if not clap_chunks: continue

                try:
                    chunk_events = await detector.predict_events_dynamic(
                        clap_chunks, sample_rate=sample_rate, top_k=2, threshold=0.15
                    )
                    chunk_embeddings = await detector.get_embeddings_batch(
                        clap_chunks, sample_rate=sample_rate
                    )
                except Exception as e:
                    logger.warning(f"Audio detection chunk {chunk_idx} failed: {e}")
                    continue

                for (window_audio, window_start), events, embedding in zip(clap_chunks, chunk_events, chunk_embeddings):
                    if not events: continue
                    window_end = window_start + clap_window
                    
                    for event in events:
                        if self._is_duplicate_event(event, window_start, previous_events, overlap_seconds):
                            continue

                        self.db.insert_audio_event(
                            media_path=str(path),
                            event_type=event["event"],
                            start_time=window_start,
                            end_time=window_end,
                            confidence=event["confidence"],
                            clap_embedding=embedding,
                        )
                        previous_events.append({"event": event["event"], "start_time": window_start})

                del audio_chunk
                gc.collect()

            detector.cleanup()

        except Exception as e:
            logger.error(f"Audio event detection failed: {e}")
            traceback.print_exc()

    async def detect_audio_language_with_confidence(self, path: Path, start_offset: float = 0.0, duration: float = 30.0) -> tuple[str, float]:
        """Detect language with confidence."""
        wav_path = None
        try:
             with AudioTranscriber() as transcriber:
                 wav_path = await transcriber._slice_audio(path, start=start_offset, end=start_offset + duration)
             return await asyncio.to_thread(self._run_detection_with_confidence, wav_path)
        except Exception:
            return ("en", 0.0)
        finally:
            if wav_path and isinstance(wav_path, Path) and wav_path != path and wav_path.exists():
                try: wav_path.unlink()
                except Exception: pass

    def _run_detection_with_confidence(self, wav_input: Path | bytes) -> tuple[str, float]:
        """Sync helper for detection."""
        import io
        with AudioTranscriber() as transcriber:
            try:
                # Assuming default model name in AudioTranscriber
                model_id = "Systran/faster-whisper-base" 
                # Note: Private method access on AudioTranscriber is dubious but kept as per original
                if model_id != transcriber._SHARED_SIZE:
                    transcriber._load_model(model_id)
                
                if transcriber._SHARED_MODEL is None: return ("en", 0.0)

                input_file = io.BytesIO(wav_input) if isinstance(wav_input, bytes) else str(wav_input)
                _, info = transcriber._SHARED_MODEL.transcribe(input_file, task="transcribe", beam_size=5)
                
                lang = info.language or "en"
                conf = info.language_probability or 0.0
                
                indic_langs = ["ta", "hi", "te", "ml", "kn", "bn", "gu", "mr", "or", "pa"]
                if lang in indic_langs and conf > 0.2:
                    conf = min(conf * 1.5, 0.95)
                    
                return (lang, conf)
            except Exception:
                return ("en", 0.0)

    async def _extract_audio_segment(self, path: Path, start: float, end: float, sample_rate: int = 48000) -> np.ndarray | None:
        try:
            cmd = [
                "ffmpeg", "-y", "-v", "error", "-ss", str(start), "-t", str(end - start),
                "-i", str(path), "-ar", str(sample_rate), "-f", "f32le", "-"
            ]
            result = subprocess.run(cmd, capture_output=True, timeout=60)
            if result.returncode != 0: return None
            return np.frombuffer(result.stdout, dtype=np.float32)
        except Exception:
            return None

    def _prepare_segments_for_db(self, *, path: Path, chunks: Iterable[dict[str, Any]]) -> list[dict[str, Any]]:
        prepared = []
        for chunk in chunks:
            text = (chunk.get("text") or "").strip()
            if not text: continue
            start = chunk.get("start")
            end = chunk.get("end") or (float(start) + 2.0)
            
            prepared.append({
                "media_path": str(path),
                "text": text,
                "start_time": float(start),
                "end_time": float(end),
                "metadata": {
                    "confidence": chunk.get("confidence", 1.0),
                    "speaker": chunk.get("speaker")
                }
            })
        return prepared

    def _is_duplicate_event(self, event: dict, event_time: float, previous_events: list, overlap_window: float) -> bool:
         for prev in previous_events:
             if prev["event"] == event["event"] and abs(prev["start_time"] - event_time) < overlap_window:
                 return True
         return False

    def _cleanup_memory(self, context: str = "") -> None:
        gc.collect()
        from core.utils.device import empty_cache
        empty_cache()
