"""Secure, modular Search Pipeline for retrieving media content."""

from typing import Any
import asyncio
from pydantic import BaseModel

from core.errors import MediaIndexerError
from core.domain.schemas import ParsedQuery
from core.utils.logger import log
from core.security.sanitizer import query_sanitizer

class SanitizerNode:
    """Node 1: Sanitizes input query using heuristics and LLM intent scoring."""
    def __init__(self, llm):
        self.llm = llm

    async def process(self, query: str) -> str:
        log("[SearchPipeline] Executing Sanitizer Node...")
        return await query_sanitizer.sanitize(query, self.llm)


class DecomposerNode:
    """Node 2: Breaks down the sanitized query into structured constraints."""
    def __init__(self, llm):
        self.llm = llm

    async def process(self, query: str) -> ParsedQuery:
        log("[SearchPipeline] Executing Decomposer Node...")
        from core.utils.prompt_loader import load_prompt
        # Using the dynamic query decomposition prompt
        prompt_template = load_prompt("dynamic_query")
        prompt = prompt_template.format(query=query)
        
        try:
            parsed = await asyncio.wait_for(
                self.llm.generate_structured(
                    schema=ParsedQuery,
                    prompt=prompt,
                    system_prompt="You are a search query parser. Return JSON only."
                ),
                timeout=12.0
            )
            log(f"[SearchPipeline] Decomposed query into {len(getattr(parsed, 'entities', []))} entities.")
            return parsed
        except Exception as e:
            log(f"[SearchPipeline] Query decomposition failed: {e}. Falling back to basic parsing.")
            return ParsedQuery(visual_keywords=[query], raw_query=query)


class RouterWeighterNode:
    """Node 3: Determines modality routing and calculates adaptive weights."""
    def __init__(self, config=None):
        self.config = config or {}

    def process(self, parsed: ParsedQuery, query: str) -> dict[str, float]:
        log("[SearchPipeline] Executing Router & Weighter Node...")
        weights = {
            "scenes": 0.4,
            "scenelets": 0.2,
            "frames": 0.2,
            "voice": 0.0,
            "audio_events": 0.0,
            "dialogue": 0.0,
            "video_metadata": 0.2
        }

        search_text = query
        if hasattr(parsed, "to_search_text"):
            search_text = parsed.to_search_text() or query

        # Boost audio/dialogue if constraints exist
        has_audio = getattr(parsed, "audio", None)
        has_voice = any(e.entity_type.lower() in ['sound', 'voice', 'speech'] for e in getattr(parsed, "entities", [])) if hasattr(parsed, "entities") else False
        has_dialogue = "says" in search_text.lower() or "talking" in search_text.lower()

        if has_audio or has_voice:
            weights["audio_events"] = 0.4
            weights["voice"] = 0.2
            weights["scenes"] = 0.2
            weights["frames"] = 0.1

        if has_dialogue:
            weights["dialogue"] = 0.5
            weights["scenes"] = 0.2
            weights["frames"] = 0.1

        # Normalize weights
        total = sum(weights.values())
        if total > 0:
            weights = {k: v / total for k, v in weights.items()}

        log(f"[SearchPipeline] Computed adaptive weights: {weights}")
        return weights


class RetrieverNode:
    """Node 4: Executes actual vector searches against Qdrant using adaptive weights."""
    def __init__(self, db):
        self.db = db

    async def process(self, parsed: ParsedQuery, query: str, weights: dict[str, float], limit: int = 20, video_path: str | None = None) -> dict[str, list[dict]]:
        log("[SearchPipeline] Executing Retriever Node...")
        search_text = query
        if hasattr(parsed, "to_search_text"):
            search_text = parsed.to_search_text() or query

        all_results = {}

        # We will only execute searches for modalities with weight > 0
        tasks = []
        
        async def fetch_scenes():
            if weights.get("scenes", 0) > 0:
                try:
                    res = await self.db.search_scenes(query=search_text, limit=limit, search_mode="hybrid", video_path=video_path)
                    all_results["scenes"] = res
                except Exception as e:
                    log(f"[SearchPipeline] Scenes retrieval failed: {e}")
                    all_results["scenes"] = []

        async def fetch_frames():
            if weights.get("frames", 0) > 0:
                try:
                    res = await self.db.search_frames_hybrid(query=search_text, limit=limit, video_paths=[video_path] if video_path else None)
                    all_results["frames"] = res
                except Exception as e:
                    log(f"[SearchPipeline] Frames retrieval failed: {e}")
                    all_results["frames"] = []

        async def fetch_audio():
            if weights.get("audio_events", 0) > 0:
                try:
                    res = await self.db.search_audio_events_semantic(query=search_text, limit=limit, video_path=video_path)
                    all_results["audio_events"] = res
                except Exception as e:
                    log(f"[SearchPipeline] Audio retrieval failed: {e}")
                    all_results["audio_events"] = []
                    
        async def fetch_dialogue():
            if weights.get("dialogue", 0) > 0:
                try:
                    res = await self.db.search_dialogue(query=search_text, limit=limit, video_path=video_path)
                    all_results["dialogue"] = res
                except Exception as e:
                    log(f"[SearchPipeline] Dialogue retrieval failed: {e}")
                    all_results["dialogue"] = []

        # Wait for all retrievers to finish
        await asyncio.gather(
            fetch_scenes(),
            fetch_frames(),
            fetch_audio(),
            fetch_dialogue()
        )
        
        # Note: Graph DB logic has been explicitly removed for Postgres migration.

        total_retrieved = sum(len(res) for res in all_results.values())
        log(f"[SearchPipeline] Retrieved {total_retrieved} total results across modalities.")
        return all_results


class SynthesizerNode:
    """Node 5: Fuses the retrieved results and prepares for Cross-Encoder Reranker integration."""
    def __init__(self):
        # We can integrate the ResultProcessorMixin or custom RRF logic here
        pass

    def process(self, results: dict[str, list[dict]], weights: dict[str, float], limit: int) -> list[dict]:
        log("[SearchPipeline] Executing Synthesizer Node...")
        
        # 1. Min-max normalization per modality
        for modality, mod_results in results.items():
            if not mod_results:
                continue
            scores = [r.get("score", 0) for r in mod_results if r.get("score") is not None]
            if not scores:
                continue
            min_s, max_s = min(scores), max(scores)
            score_range = max_s - min_s
            for r in mod_results:
                raw = r.get("score", 0)
                r["_raw_score"] = raw
                if score_range > 0:
                    r["score"] = (raw - min_s) / score_range
                else:
                    r["score"] = 1.0

        # 2. Simple fusion using weights
        fused = {}
        for modality, mod_results in results.items():
            weight = weights.get(modality, 0.1)
            for r in mod_results:
                rid = r.get("id")
                if not rid:
                    continue
                weighted_score = r["score"] * weight
                if rid not in fused:
                    fused[rid] = r.copy()
                    fused[rid]["fused_score"] = weighted_score
                    fused[rid]["matched_modalities"] = [modality]
                else:
                    fused[rid]["fused_score"] += weighted_score
                    if modality not in fused[rid]["matched_modalities"]:
                        fused[rid]["matched_modalities"].append(modality)

        candidates = list(fused.values())
        candidates.sort(key=lambda x: x.get("fused_score", 0), reverse=True)
        candidates = candidates[:limit]

        log(f"[SearchPipeline] Synthesized {len(candidates)} fused results.")
        return candidates


class SearchPipeline:
    """Central orchestrator for the decoupled search state-machine/pipeline."""
    def __init__(self, db, llm):
        self.db = db
        self.llm = llm
        
        self.sanitizer = SanitizerNode(llm=self.llm)
        self.decomposer = DecomposerNode(llm=self.llm)
        self.router = RouterWeighterNode()
        self.retriever = RetrieverNode(db=self.db)
        self.synthesizer = SynthesizerNode()

    async def execute(self, query: str, limit: int = 20, video_path: str | None = None) -> dict[str, Any]:
        """Executes the strict, secure pipeline pattern."""
        log(f"[SearchPipeline] Starting execution for query: '{query[:50]}...'")
        
        try:
            # Step 1: Sanitization (Security First)
            sanitized_query = await self.sanitizer.process(query)
            
            # Step 2: Decomposition
            parsed_query = await self.decomposer.process(sanitized_query)
            
            # Step 3: Routing & Adaptive Weighting
            weights = self.router.process(parsed_query, sanitized_query)
            
            # Step 4: Retrieval (Graph DB logic removed)
            retrieval_results = await self.retriever.process(
                parsed_query, sanitized_query, weights, limit=limit*2, video_path=video_path
            )
            
            # Step 5: Synthesis
            final_results = self.synthesizer.process(retrieval_results, weights, limit)
            
            return {
                "query": query,
                "sanitized": True,
                "parsed": parsed_query.model_dump() if hasattr(parsed_query, "model_dump") else {},
                "results": final_results,
                "result_count": len(final_results)
            }
            
        except MediaIndexerError as e:
            log(f"[SearchPipeline] Pipeline aborted due to explicit error: {e}")
            raise
        except Exception as e:
            log(f"[SearchPipeline] Unhandled pipeline failure: {e}")
            raise MediaIndexerError(f"Search pipeline failed: {e}", original_error=e)
