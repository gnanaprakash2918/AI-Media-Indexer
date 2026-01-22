"""Reranking Council for search result verification.

Prompts loaded from external files.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING

from pydantic import BaseModel

from core.llm.vlm_factory import VLMClient, get_vlm_client
from core.processing.scene_detector import extract_scene_frame
from core.utils.logger import get_logger
from core.utils.prompt_loader import load_prompt


log = get_logger(__name__)


class RerankScore(BaseModel):
    """Structured output for VLM re-ranking results."""

    confidence: int = 0
    reason: str = ""


@dataclass
class SearchCandidate:
    """Represents a potential search match before result calibration."""

    video_path: str
    start_time: float
    end_time: float
    score: float
    payload: dict = field(default_factory=dict)


@dataclass
class RankedResult:
    """A search result scored by the reranking council."""

    candidate: "SearchCandidate"
    final_score: float = 0.0
    vlm_confidence: int = 0
    vlm_reason: str = ""
    cross_encoder_score: float = 0.0
    bge_score: float = 0.0
    sources: dict = field(default_factory=dict)


# Load from external file
RERANK_PROMPT = load_prompt("rerank_frames")


class RerankingCouncil:
    """Multi-model reranking council per AGENTS.MD.

    Combines:
    - Cross-encoder (ms-marco-MiniLM) for query-document relevance
    - BGE-Reranker v2 for semantic scoring
    - VLM analysis for visual verification

    Usage:
        council = RerankingCouncil(weights=(0.35, 0.35, 0.30))
        results = await council.council_rerank(query, candidates)
    """

    def __init__(
        self,
        client: VLMClient | None = None,
        weights: tuple[float, float, float] = (0.35, 0.35, 0.30),
    ):
        """Initialize reranking council.

        Args:
            client: Optional VLM client for visual verification.
            weights: (cross_encoder, bge, vlm) weights for fusion.
        """
        self._client = client
        self.weights = weights
        self._cross_encoder = None
        self._bge_reranker = None
        self._colbert = None  # Late Interaction
        self._models_loaded = False

    @property
    def client(self) -> VLMClient:
        """Lazy load the VLM client."""
        if self._client is None:
            self._client = get_vlm_client()
        return self._client

    def _lazy_load_models(self) -> None:
        """Load cross-encoder and BGE models lazily."""
        if self._models_loaded:
            return

        try:
            from sentence_transformers import CrossEncoder

            # UPGRADED per Council: L-12 provides deeper reasoning than L-6
            self._cross_encoder = CrossEncoder(
                "cross-encoder/ms-marco-MiniLM-L-12-v2",
                max_length=512,
            )
            log.info("[RerankCouncil] Loaded cross-encoder (L-12-v2)")
        except ImportError:
            log.warning("[RerankCouncil] sentence-transformers not available")
        except Exception as e:
            log.warning(f"[RerankCouncil] Cross-encoder load failed: {e}")

        try:
            from FlagEmbedding import FlagReranker

            self._bge_reranker = FlagReranker(
                "BAAI/bge-reranker-v2-m3",
                use_fp16=True,
            )
            log.info("[RerankCouncil] Loaded BGE-Reranker")
        except ImportError:
            log.warning("[RerankCouncil] FlagEmbedding not available")
        except Exception as e:
            log.warning(f"[RerankCouncil] BGE-Reranker load failed: {e}")

        # Load ColBERT (Late Interaction)
        try:
            from core.retrieval.late_interaction import ColBERTRetriever

            self._colbert = ColBERTRetriever()
            # Note: ColBERTRetriever is also lazy-loaded internally on first use
            log.info("[RerankCouncil] Initialized ColBERT Retriever")
        except Exception as e:
            log.warning(f"[RerankCouncil] ColBERT init failed: {e}")

        self._models_loaded = True

    def _get_text_for_ranking(self, candidate: "SearchCandidate") -> str:
        """Extract ALL multimodal text from candidate for comprehensive ranking.

        This method extracts text from ALL available fields to ensure the
        cross-encoder and BGE reranker have complete context for scoring.

        Fields extracted (in priority order):
        1. Scene-level multimodal texts (visual_text, motion_text, dialogue_text)
        2. Identity information (person_names, face_names, speaker_name)
        3. Actions and sequences
        4. OCR/visible text
        5. Location and context
        6. Clothing and accessories
        7. Description fallbacks
        """
        payload = candidate.payload or {}
        parts = []

        # === 1. Scene-level multimodal texts (PRIMARY - most complete) ===
        if payload.get("visual_text"):
            parts.append(f"Visual: {payload['visual_text']}")
        if payload.get("motion_text"):
            parts.append(f"Actions: {payload['motion_text']}")
        if payload.get("dialogue_text"):
            parts.append(f"Dialogue: {payload['dialogue_text']}")
        if payload.get("dialogue_transcript"):
            parts.append(f"Transcript: {payload['dialogue_transcript'][:200]}")

        # === 2. Identity information (CRITICAL for person queries) ===
        if payload.get("person_names"):
            names = payload["person_names"]
            if isinstance(names, list):
                names = [n for n in names if n]  # Filter empty
                if names:
                    parts.append(f"People: {', '.join(names)}")
            elif names:
                parts.append(f"People: {names}")

        if payload.get("face_names"):
            names = payload["face_names"]
            if isinstance(names, list):
                names = [n for n in names if n]
                if names:
                    parts.append(f"Faces: {', '.join(names)}")
            elif names:
                parts.append(f"Faces: {names}")

        if payload.get("speaker_name"):
            parts.append(f"Speaker: {payload['speaker_name']}")
        if payload.get("speaker_names"):
            names = payload["speaker_names"]
            if isinstance(names, list) and names:
                parts.append(
                    f"Speakers: {', '.join(str(n) for n in names if n)}"
                )

        # === 3. Actions (critical for queries like "dancing" or "running") ===
        if payload.get("actions"):
            actions = payload["actions"]
            if isinstance(actions, list) and actions:
                parts.append(f"Actions: {', '.join(str(a) for a in actions)}")
            elif actions:
                parts.append(f"Actions: {actions}")

        if payload.get("action_sequence"):
            parts.append(f"Sequence: {payload['action_sequence']}")

        # === 4. OCR/Visible Text (for brand/name queries) ===
        if payload.get("visible_text"):
            text_items = payload["visible_text"]
            if isinstance(text_items, list) and text_items:
                parts.append(
                    f"Text: {', '.join(str(t) for t in text_items[:5])}"
                )
            elif text_items:
                parts.append(f"Text: {text_items}")

        if payload.get("ocr_text"):
            ocr = payload["ocr_text"]
            if isinstance(ocr, list) and ocr:
                parts.append(f"OCR: {', '.join(str(o) for o in ocr[:5])}")
            elif ocr:
                parts.append(f"OCR: {ocr}")

        # === 5. Location and context ===
        if payload.get("location"):
            parts.append(f"Location: {payload['location']}")
        if payload.get("scene_location"):
            parts.append(f"Setting: {payload['scene_location']}")
        if payload.get("cultural_context"):
            parts.append(f"Context: {payload['cultural_context']}")

        # === 6. Clothing and accessories (for appearance queries) ===
        if payload.get("clothing_colors"):
            colors = payload["clothing_colors"]
            if isinstance(colors, list) and colors:
                parts.append(
                    f"Clothing colors: {', '.join(str(c) for c in colors)}"
                )

        if payload.get("clothing_types"):
            types = payload["clothing_types"]
            if isinstance(types, list) and types:
                parts.append(f"Clothing: {', '.join(str(t) for t in types)}")

        if payload.get("accessories"):
            acc = payload["accessories"]
            if isinstance(acc, list) and acc:
                parts.append(f"Accessories: {', '.join(str(a) for a in acc)}")

        # === 7. Entity names (objects, brands) ===
        if payload.get("entity_names"):
            entities = payload["entity_names"]
            if isinstance(entities, list) and entities:
                parts.append(f"Entities: {', '.join(str(e) for e in entities)}")

        # === 8. Description fallbacks ===
        if payload.get("visual_summary"):
            parts.append(f"Summary: {payload['visual_summary'][:200]}")
        if payload.get("description"):
            parts.append(f"Description: {payload['description'][:200]}")
        if payload.get("dense_caption"):
            parts.append(f"Caption: {payload['dense_caption'][:200]}")
        if payload.get("scene_description"):
            parts.append(f"Scene: {payload['scene_description'][:200]}")

        # Combine all parts
        full_text = " | ".join(parts)

        # If still empty, try legacy single-field extraction
        if not full_text:
            full_text = (
                payload.get("description", "")
                or payload.get("dense_caption", "")
                or payload.get("transcript", "")
                or payload.get("text", "")
                or "unknown scene"
            )

        # Limit length for models (cross-encoder has 512 token limit)
        return str(full_text)[:2000]

    async def council_rerank(
        self,
        query: str,
        candidates: list["SearchCandidate"],
        max_candidates: int = 10,
        use_vlm: bool = True,
    ) -> list[RankedResult]:
        """Rerank using full council with weighted RRF fusion.

        Args:
            query: Search query.
            candidates: Candidates to rerank.
            max_candidates: Max candidates to process.
            use_vlm: Whether to use VLM visual verification.

        Returns:
            List of RankedResult sorted by final_score.
        """
        self._lazy_load_models()
        candidates = candidates[:max_candidates]
        scores: dict[int, dict] = defaultdict(
            lambda: {
                "cross": 0.0,
                "bge": 0.0,
                "vlm": 0.0,
                "candidate": None,
                "vlm_reason": "",
            }
        )

        # Cross-encoder scoring
        if self._cross_encoder:
            try:
                pairs = [
                    (query, self._get_text_for_ranking(c)) for c in candidates
                ]
                ce_scores = self._cross_encoder.predict(list(pairs))
                for i, score in enumerate(ce_scores):
                    scores[i]["cross"] = float(score)
                    scores[i]["candidate"] = candidates[i]
            except Exception as e:
                log.warning(f"[RerankCouncil] Cross-encoder failed: {e}")

        # BGE-Reranker scoring
        if self._bge_reranker:
            try:
                pairs = [
                    (query, self._get_text_for_ranking(c)) for c in candidates
                ]
                bge_scores = self._bge_reranker.compute_score(pairs)
                if bge_scores is not None:
                    if isinstance(bge_scores, (int, float)):
                        bge_scores = [bge_scores]
                    for i, score in enumerate(bge_scores):
                        scores[i]["bge"] = float(score)
                        scores[i]["candidate"] = candidates[i]
            except Exception as e:
                log.warning(f"[RerankCouncil] BGE-Reranker failed: {e}")

        # VLM visual verification (optional, expensive)
        if use_vlm:
            prompt = RERANK_PROMPT.format(query=query)
            for i, candidate in enumerate(candidates):
                vlm_score = self._score_candidate(candidate, prompt)
                scores[i]["vlm"] = vlm_score.confidence / 100.0
                scores[i]["vlm_reason"] = vlm_score.reason
                scores[i]["candidate"] = candidate

        # Weighted fusion with HITL feedback integration
        w_ce, w_bge, w_vlm = self.weights
        results = []

        # Lazy load HITL feedback manager
        try:
            from core.retrieval.hitl_feedback import get_hitl_manager

            hitl = get_hitl_manager()
        except ImportError:
            hitl = None
            log.debug("[RerankCouncil] HITL feedback not available")

        # ColBERT scoring (Late Interaction)
        if self._colbert:
            try:
                # Encode query once
                q_enc = await self._colbert.encode_query(query)
                if q_enc:
                    # Encode all candidates (batch)
                    texts = [self._get_text_for_ranking(c) for c in candidates]
                    d_encs = await self._colbert.encode_documents(texts)

                    for i, d_enc in enumerate(d_encs):
                        # Compute MaxSim score
                        score = self._colbert.compute_score(
                            q_enc["colbert_vecs"][0],  # Query has 1 item
                            d_enc["colbert_vecs"],
                        )
                        scores[i]["colbert"] = score
                        scores[i]["candidate"] = candidates[i]
            except Exception as e:
                log.warning(f"[RerankCouncil] ColBERT failed: {e}")

        for _i, data in scores.items():
            if data["candidate"] is None:
                continue

            # Normalize scores (approximate ranges)
            # Cross: 0..1 (logits->sigmoid often 0..1, but here it's MiniLM raw logits? No, CrossEncoder usually returns logits or 0-1 depending on usage)
            # BGE: often negative to positive. Need normalization?
            # For now, we assume raw scores roughly align or use weights to compensate.
            # ColBERT sequences are usually sums of dot products -> can be large (10-30). Need normalization.

            s_cross = data["cross"]
            s_bge = data["bge"]
            s_vlm = data["vlm"]
            s_colbert = data.get("colbert", 0.0)

            # Simple normalization for ColBERT (heuristic)
            s_colbert_norm = min(
                max(s_colbert / 32.0, 0.0), 1.0
            )  # Approx range 0..32 -> 0..1

            # Calculate base score from council
            # New Weights with ColBERT: (0.25, 0.25, 0.25, 0.25) or similar
            # If ColBERT missing, redistribute

            if self._colbert:
                base_score = (
                    0.25 * s_cross
                    + 0.25 * s_bge
                    + 0.25 * s_colbert_norm
                    + 0.25 * s_vlm
                )
            else:
                base_score = w_ce * s_cross + w_bge * s_bge + w_vlm * s_vlm

            # Apply HITL boost/penalty if available
            hitl_boost = 1.0
            if hitl:
                try:
                    hitl_boost = hitl.get_boost_factor(
                        query=query,
                        video_path=data["candidate"].video_path,
                        timestamp=data["candidate"].start_time,
                    )
                except Exception as e:
                    log.debug(f"[RerankCouncil] HITL boost failed: {e}")

            final = base_score * hitl_boost

            results.append(
                RankedResult(
                    candidate=data["candidate"],
                    final_score=final,
                    vlm_confidence=int(data["vlm"] * 100),
                    vlm_reason=data["vlm_reason"],
                    cross_encoder_score=data["cross"],
                    bge_score=data["bge"],
                    sources={
                        "cross_encoder": data["cross"],
                        "bge_reranker": data["bge"],
                        "colbert": s_colbert,
                        "vlm": data["vlm"],
                        "hitl_boost": hitl_boost,
                    },
                )
            )

        results.sort(key=lambda r: r.final_score, reverse=True)
        log.info(
            f"[RerankCouncil] Reranked {len(results)} candidates, "
            f"ColBERT={'active' if self._colbert else 'inactive'}, HITL={'enabled' if hitl else 'disabled'}"
        )
        return results

    def rerank(
        self,
        query: str,
        candidates: list[SearchCandidate],
        max_candidates: int = 20,
    ) -> list[RankedResult]:
        """Re-rank candidates based on VLM visual analysis."""
        results: list[RankedResult] = []
        prompt = RERANK_PROMPT.format(query=query)

        for candidate in candidates[:max_candidates]:
            score = self._score_candidate(candidate, prompt)
            results.append(
                RankedResult(
                    candidate=candidate,
                    vlm_confidence=score.confidence,
                    vlm_reason=score.reason,
                )
            )

        results.sort(key=lambda r: r.vlm_confidence, reverse=True)
        return results

    def _score_candidate(
        self, candidate: SearchCandidate, prompt: str
    ) -> RerankScore:
        """Score a candidate using VLM visual analysis with temporal context.

        Uses multiple frames (start, middle, end) for temporal understanding
        rather than just a single middle frame.
        """
        frames = self._extract_keyframes(candidate)
        if not frames:
            return RerankScore(confidence=0, reason="No frames extracted")

        # Select frames for analysis: start, middle, end for temporal context
        if len(frames) >= 3:
            frames_to_analyze = [
                frames[0],
                frames[len(frames) // 2],
                frames[-1],
            ]
        else:
            frames_to_analyze = frames

        # Build temporal context into prompt
        duration = candidate.end_time - candidate.start_time
        temporal_prompt = (
            f"{prompt}\n\n"
            f"TEMPORAL CONTEXT: This is a {duration:.1f}-second video segment "
            f"from {candidate.start_time:.1f}s to {candidate.end_time:.1f}s. "
            f"You are seeing {len(frames_to_analyze)} frames representing the start, middle, and end. "
            f"Consider the progression and movement across frames when scoring."
        )

        try:
            # Try multi-image analysis first (if VLM supports it)
            if len(frames_to_analyze) > 1:
                try:
                    # Some VLMs support multiple images in one call
                    if hasattr(
                        self.client, "generate_caption_from_bytes_multi"
                    ):
                        raw = self.client.generate_caption_from_bytes_multi(
                            frames_to_analyze, temporal_prompt
                        )
                        if raw:
                            return self._parse_vlm_response(raw)
                except (AttributeError, NotImplementedError):
                    pass

            # Fallback: Analyze each frame and aggregate scores
            frame_scores = []
            frame_reasons = []

            for i, frame in enumerate(frames_to_analyze):
                frame_label = (
                    ["start", "middle", "end"][i]
                    if len(frames_to_analyze) == 3
                    else f"frame_{i}"
                )
                frame_prompt = (
                    f"{prompt}\n\n(Analyzing {frame_label} of the segment)"
                )

                raw = self.client.generate_caption_from_bytes(
                    frame, frame_prompt
                )
                if raw:
                    score = self._parse_vlm_response(raw)
                    if score.confidence > 0:
                        frame_scores.append(score.confidence)
                        frame_reasons.append(f"{frame_label}: {score.reason}")

            if not frame_scores:
                return RerankScore(
                    confidence=0, reason="VLM analysis failed for all frames"
                )

            # Aggregate: Use max confidence but penalize if frames disagree significantly
            max_conf = max(frame_scores)
            min_conf = min(frame_scores)
            avg_conf = sum(frame_scores) / len(frame_scores)

            # If frames have consistent scores, trust the max
            # If they disagree a lot, use average to be conservative
            if max_conf - min_conf > 30:
                final_conf = int(avg_conf)
                reason = f"Mixed signals across {len(frame_scores)} frames: {', '.join(frame_reasons[:2])}"
            else:
                final_conf = int(max_conf)
                reason = f"Consistent across {len(frame_scores)} frames: {frame_reasons[0] if frame_reasons else 'no detail'}"

            return RerankScore(confidence=final_conf, reason=reason)

        except Exception as e:
            log.error(f"VLM rerank error: {e}")
            return RerankScore(confidence=0, reason=str(e)[:50])

    def _parse_vlm_response(self, raw: str) -> RerankScore:
        """Parse VLM response into RerankScore."""
        if not raw:
            return RerankScore(confidence=0, reason="VLM returned empty")

        clean = raw.strip()
        if clean.startswith("```"):
            clean = clean.split("```")[1]
            if clean.startswith("json"):
                clean = clean[4:]

        try:
            # 1. Try standard JSON validation
            if clean.startswith("```json"):
                clean = clean.split("```json")[1]
            if clean.startswith("```"):
                clean = clean.split("```")[1]
            if clean.endswith("```"):
                clean = clean.rsplit("```", 1)[0]

            clean = clean.strip()
            return RerankScore.model_validate_json(clean)
        except Exception:
            # 2. Try heuristic extraction if JSON fails
            try:
                import re

                conf_match = re.search(r'"confidence":\s*(\d+)', clean)
                reason_match = re.search(r'"reason":\s*"([^"]+)"', clean)

                if conf_match:
                    conf = int(conf_match.group(1))
                    reason = (
                        reason_match.group(1)
                        if reason_match
                        else "Parsed from partial"
                    )
                    return RerankScore(confidence=conf, reason=reason)
            except Exception:
                pass

            # 3. Fallback: If we got text but not JSON
            if len(clean) > 10:
                return RerankScore(
                    confidence=50, reason=f"Unstructured: {clean[:50]}..."
                )

            return RerankScore(confidence=0, reason="Parse failed")

    def _extract_keyframes(self, candidate: SearchCandidate) -> list[bytes]:
        frames = []
        path = Path(candidate.video_path)
        if not path.exists():
            return frames

        duration = candidate.end_time - candidate.start_time

        # Adaptive sampling: More frames for longer clips to catch dynamic actions
        # Min 3 frames (start, mid, end) for short clips
        # Max 8 frames for long clips to prevent OOM/timeouts
        # Stride ~3-5 seconds ideally

        num_frames = 3
        if duration > 5.0:
            num_frames = min(8, int(duration / 3.0))
            num_frames = max(3, num_frames)

        step = duration / (num_frames - 1) if num_frames > 1 else 0
        timestamps = [
            candidate.start_time + i * step for i in range(num_frames)
        ]

        # Ensure last frame is slightly before end to avoid seeking past EOF
        timestamps[-1] = max(candidate.start_time, candidate.end_time - 0.1)

        for ts in timestamps:
            frame_bytes = extract_scene_frame(path, ts)
            if frame_bytes:
                frames.append(frame_bytes)

        return frames


# Backwards compatibility alias
VLMReranker = RerankingCouncil
