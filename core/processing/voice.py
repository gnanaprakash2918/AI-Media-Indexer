"""Voice processing module for speaker diarization and embedding extraction."""

from __future__ import annotations

import asyncio
from pathlib import Path
from typing import Final, cast

import numpy as np
import torch
from pyannote.audio import Inference, Model, Pipeline
from pyannote.core import Segment

from config import settings
from core.schemas import SpeakerSegment
from core.utils.logger import log

GPU_SEMAPHORE = asyncio.Semaphore(1)

MIN_SEGMENT_DURATION: Final = 0.8
MAX_SEGMENT_DURATION: Final = 30.0
EMBEDDING_VERSION: Final = "wespeaker_resnet34_v1_l2"


class VoiceProcessor:
    """Handles speaker diarization and voice embedding extraction."""

    def __init__(self) -> None:
        """Initialize the voice processor with settings from config."""
        self.enabled = bool(settings.enable_voice_analysis)
        self.device = torch.device(settings.device)
        self.hf_token = settings.hf_token if settings.hf_token else None

        self.pipeline: Pipeline | None = None
        self.embedding_model: Model | None = None
        self.inference: Inference | None = None
        self._initialized = False
        self._init_lock = asyncio.Lock()

        if not self.enabled:
            log.info("Voice analysis disabled by config.")
            return

        if not self.hf_token:
            log.warning("HF_TOKEN missing. Voice analysis disabled.")
            self.enabled = False

    async def _lazy_init(self) -> None:
        if not self.enabled or self._initialized:
            return

        async with self._init_lock:
            if self._initialized:
                return

            try:
                async with GPU_SEMAPHORE:
                    log.info(
                        f"Loading pyannote pipeline={settings.pyannote_model} "
                        f"embedder={settings.voice_embedding_model} "
                        f"device={self.device}"
                    )

                    self.pipeline = Pipeline.from_pretrained(
                        settings.pyannote_model,
                        use_auth_token=self.hf_token,
                    )
                    self.pipeline.to(self.device)

                    self.embedding_model = Model.from_pretrained(
                        settings.voice_embedding_model,
                        use_auth_token=self.hf_token,
                    )

                    self.inference = Inference(
                        self.embedding_model,
                        window="whole",
                        device=self.device,
                    )

                self._initialized = True

            except Exception as e:
                log.error(f"Voice model initialization failed: {e}")
                self.enabled = False
                self.pipeline = None
                self.embedding_model = None
                self.inference = None

    async def process(self, audio_path: Path) -> list[SpeakerSegment]:
        """Process an audio file to extract speaker segments and embeddings.

        Args:
            audio_path: Path to the audio file.

        Returns:
            List of detected speaker segments.
        """
        if not self.enabled or not audio_path.exists():
            return []

        await self._lazy_init()
        if not self.pipeline or not self.inference:
            return []

        segments: list[SpeakerSegment] = []

        try:
            async with GPU_SEMAPHORE:
                diarization = self.pipeline(
                    str(audio_path),
                    min_speakers=settings.min_speakers,
                    max_speakers=settings.max_speakers,
                )

            for turn, _, speaker in diarization.itertracks(yield_label=True):
                start = float(turn.start)
                end = float(turn.end)
                duration = end - start

                if duration < MIN_SEGMENT_DURATION:
                    continue
                if duration > MAX_SEGMENT_DURATION:
                    end = start + MAX_SEGMENT_DURATION

                embedding = await self._extract_embedding(audio_path, start, end)
                if embedding is None:
                    continue

                segments.append(
                    SpeakerSegment(
                        start_time=start,
                        end_time=end,
                        speaker_label=speaker,
                        confidence=1.0,
                        embedding=embedding,
                    )
                )

            return segments

        except Exception as e:
            log.error(f"Voice processing failed for {audio_path.name}: {e}")
            return []

