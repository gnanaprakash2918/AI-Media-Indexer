"""Hierarchical Video Summarizer.

Generates multi-level summaries for improved RAG:
- L1: Full video summary ("What is the plot?")
- L2: Scene/chapter summaries (5-minute chunks)

Inspired by RAPTOR and Ragie's Summary Index approach.
"""

from __future__ import annotations

import asyncio
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field

from config import settings
from core.storage.db import VectorDB
from core.utils.logger import log
from llm.interface import LLMInterface, get_llm

SCENE_SUMMARY_PROMPT = """Summarize this 5-minute scene from a video.

TRANSCRIPT:
{transcript}

VISUAL DESCRIPTIONS:
{visual_descriptions}

KEY ENTITIES:
{entities}

Provide a concise summary (3-5 sentences) covering:
1. Main actions and events
2. Key dialogue or spoken content
3. Notable visual elements
4. Any emotional tone or atmosphere

SUMMARY:"""

VIDEO_SUMMARY_PROMPT = """Create a comprehensive summary of this entire video.

SCENE SUMMARIES:
{scene_summaries}

VIDEO METADATA:
- Title: {title}
- Duration: {duration}
- Total Scenes: {scene_count}

Write a cohesive summary (1-2 paragraphs) that:
1. Describes the overall narrative or content
2. Highlights key themes and main characters/subjects
3. Notes the most significant moments
4. Captures the tone and style

SUMMARY:"""


class SummaryLevel:
    L1_VIDEO = "l1_video"
    L2_SCENE = "l2_scene"


class VideoSummary(BaseModel):
    video_path: str
    level: str
    summary: str
    start_time: float = 0.0
    end_time: float = 0.0
    scene_index: int | None = None
    entities: list[str] = Field(default_factory=list)
    embedding: list[float] | None = None


class HierarchicalSummarizer:
    """Generates hierarchical summaries for video content.

    Creates two levels of summaries:
    - L2 (Scene): 5-minute chunks with local context
    - L1 (Video): Full video summary from L2 aggregation
    """

    SCENE_DURATION_SECONDS = 300  # 5 minutes

    def __init__(
        self,
        db: VectorDB | None = None,
        llm: LLMInterface | None = None,
    ):
        self.db = db or VectorDB()
        self.llm = llm or get_llm()
        self._scene_duration = getattr(
            settings, "summary_scene_duration", self.SCENE_DURATION_SECONDS
        )

    async def summarize_video(
        self,
        video_path: str,
        force: bool = False,
    ) -> dict[str, Any]:
        """Generate hierarchical summaries for a video.

        Args:
            video_path: Path to the video file.
            force: If True, regenerate even if summaries exist.

        Returns:
            Dict with l1_summary and l2_summaries.
        """
        # Check if summaries already exist (lazy)
        if not force:
            existing = self._get_existing_summaries(video_path)
            if existing.get("l1_summary"):
                log(f"Summaries already exist for {video_path}, skipping")
                return existing

        log(f"Generating hierarchical summaries for {video_path}")

        # Get all indexed frames/segments for this video
        frames = self._get_video_frames(video_path)
        if not frames:
            log(f"No indexed frames found for {video_path}")
            return {"l1_summary": None, "l2_summaries": []}

        # Get video duration
        duration = max(f.get("timestamp", 0) for f in frames)

        # Generate L2 (scene) summaries
        l2_summaries = await self._generate_scene_summaries(
            video_path, frames, duration
        )

        # Generate L1 (video) summary from L2s
        l1_summary = await self._generate_video_summary(
            video_path, l2_summaries, duration
        )

        # Store summaries in Qdrant
        await self._store_summaries(video_path, l1_summary, l2_summaries)

        return {
            "l1_summary": l1_summary,
            "l2_summaries": l2_summaries,
            "scene_count": len(l2_summaries),
        }

    def _get_existing_summaries(self, video_path: str) -> dict[str, Any]:
        """Check for existing summaries in the database."""
        try:
            results = self.db.client.scroll(
                collection_name=self.db.SUMMARIES_COLLECTION,
                scroll_filter={
                    "must": [
                        {"key": "video_path", "match": {"value": video_path}},
                        {"key": "level", "match": {"value": SummaryLevel.L1_VIDEO}},
                    ]
                },
                limit=1,
                with_payload=True,
            )

            if results[0]:
                l1 = results[0][0].payload

                # Get L2 summaries
                l2_results = self.db.client.scroll(
                    collection_name=self.db.SUMMARIES_COLLECTION,
                    scroll_filter={
                        "must": [
                            {"key": "video_path", "match": {"value": video_path}},
                            {"key": "level", "match": {"value": SummaryLevel.L2_SCENE}},
                        ]
                    },
                    limit=100,
                    with_payload=True,
                )

                l2_summaries = [p.payload for p in l2_results[0]]
                return {"l1_summary": l1, "l2_summaries": l2_summaries}

        except Exception:
            pass

        return {"l1_summary": None, "l2_summaries": []}

    def _get_video_frames(self, video_path: str) -> list[dict[str, Any]]:
        """Get all indexed frames for a video."""
        try:
            results = self.db.client.scroll(
                collection_name=self.db.FRAMES_COLLECTION,
                scroll_filter={
                    "must": [{"key": "video_path", "match": {"value": video_path}}]
                },
                limit=10000,
                with_payload=True,
            )

            return [p.payload for p in results[0]]
        except Exception as e:
            log(f"Error getting frames: {e}")
            return []

    async def _generate_scene_summaries(
        self,
        video_path: str,
        frames: list[dict[str, Any]],
        duration: float,
    ) -> list[dict[str, Any]]:
        """Generate L2 scene summaries for 5-minute chunks."""
        summaries = []
        scene_count = max(1, int(duration / self._scene_duration) + 1)

        for i in range(scene_count):
            start_time = i * self._scene_duration
            end_time = min((i + 1) * self._scene_duration, duration)

            # Get frames in this time range
            scene_frames = [
                f for f in frames if start_time <= f.get("timestamp", 0) < end_time
            ]

            if not scene_frames:
                continue

            # Build context from frames
            transcript = self._extract_transcripts(scene_frames)
            visuals = self._extract_visuals(scene_frames)
            entities = self._extract_entities(scene_frames)

            # Generate summary via LLM
            prompt = SCENE_SUMMARY_PROMPT.format(
                transcript=transcript or "(No dialogue)",
                visual_descriptions=visuals or "(No visual descriptions)",
                entities=", ".join(entities) if entities else "(None identified)",
            )

            try:
                summary_text = await asyncio.get_event_loop().run_in_executor(
                    None, lambda: self.llm.generate(prompt, max_tokens=300)
                )

                summaries.append(
                    {
                        "video_path": video_path,
                        "level": SummaryLevel.L2_SCENE,
                        "scene_index": i,
                        "start_time": start_time,
                        "end_time": end_time,
                        "summary": summary_text.strip(),
                        "entities": entities,
                    }
                )

            except Exception as e:
                log(f"Error generating scene {i} summary: {e}")

        return summaries

    async def _generate_video_summary(
        self,
        video_path: str,
        scene_summaries: list[dict[str, Any]],
        duration: float,
    ) -> dict[str, Any]:
        """Generate L1 video summary from L2 scene summaries."""
        if not scene_summaries:
            return {
                "video_path": video_path,
                "level": SummaryLevel.L1_VIDEO,
                "summary": "No content available for summarization.",
                "start_time": 0.0,
                "end_time": duration,
            }

        # Combine scene summaries
        scene_texts = []
        all_entities = set()

        for s in scene_summaries:
            idx = s.get("scene_index", 0)
            start = s.get("start_time", 0)
            end = s.get("end_time", 0)
            scene_texts.append(
                f"Scene {idx + 1} ({int(start // 60)}:{int(start % 60):02d} - "
                f"{int(end // 60)}:{int(end % 60):02d}):\n{s.get('summary', '')}"
            )
            all_entities.update(s.get("entities", []))

        title = Path(video_path).stem

        prompt = VIDEO_SUMMARY_PROMPT.format(
            scene_summaries="\n\n".join(scene_texts),
            title=title,
            duration=f"{int(duration // 60)} minutes",
            scene_count=len(scene_summaries),
        )

        try:
            summary_text = await asyncio.get_event_loop().run_in_executor(
                None, lambda: self.llm.generate(prompt, max_tokens=500)
            )

            return {
                "video_path": video_path,
                "level": SummaryLevel.L1_VIDEO,
                "summary": summary_text.strip(),
                "start_time": 0.0,
                "end_time": duration,
                "entities": list(all_entities),
            }

        except Exception as e:
            log(f"Error generating video summary: {e}")
            return {
                "video_path": video_path,
                "level": SummaryLevel.L1_VIDEO,
                "summary": f"Video contains {len(scene_summaries)} scenes.",
                "start_time": 0.0,
                "end_time": duration,
            }

    def _extract_transcripts(self, frames: list[dict[str, Any]]) -> str:
        """Extract transcript text from frames."""
        texts = []
        for f in frames:
            if dialogue := f.get("dialogue"):
                texts.append(dialogue)
            if transcript := f.get("transcript"):
                texts.append(transcript)
        return " ".join(texts)[:2000]  # Limit length

    def _extract_visuals(self, frames: list[dict[str, Any]]) -> str:
        """Extract visual descriptions from frames."""
        visuals = []
        for f in frames:
            if desc := f.get("action"):
                visuals.append(desc)
            if visual := f.get("visual_description"):
                visuals.append(visual)
        # Deduplicate and limit
        unique = list(dict.fromkeys(visuals))
        return " | ".join(unique[:20])

    def _extract_entities(self, frames: list[dict[str, Any]]) -> list[str]:
        """Extract unique entities from frames."""
        entities = set()
        for f in frames:
            if ents := f.get("entities"):
                if isinstance(ents, list):
                    entities.update(ents)
            if faces := f.get("face_names"):
                if isinstance(faces, list):
                    entities.update(faces)
        return list(entities)[:20]

    async def _store_summaries(
        self,
        video_path: str,
        l1_summary: dict[str, Any],
        l2_summaries: list[dict[str, Any]],
    ) -> None:
        """Store summaries in the global_summaries collection."""
        import uuid

        from qdrant_client.models import PointStruct

        # Ensure collection exists
        self.db._ensure_collection(self.db.SUMMARIES_COLLECTION)

        points = []

        # Add L1 summary
        if l1_summary and l1_summary.get("summary"):
            embedding = self.db.encode_texts([l1_summary["summary"]])[0]
            points.append(
                PointStruct(
                    id=str(uuid.uuid4()),
                    vector=embedding.tolist(),
                    payload=l1_summary,
                )
            )

        # Add L2 summaries
        for s in l2_summaries:
            if s.get("summary"):
                embedding = self.db.encode_texts([s["summary"]])[0]
                points.append(
                    PointStruct(
                        id=str(uuid.uuid4()),
                        vector=embedding.tolist(),
                        payload=s,
                    )
                )

        if points:
            self.db.client.upsert(
                collection_name=self.db.SUMMARIES_COLLECTION,
                points=points,
            )
            log(f"Stored {len(points)} summaries for {video_path}")


# Global instance
summarizer = HierarchicalSummarizer()
