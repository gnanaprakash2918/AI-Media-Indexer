"""Agent orchestration for media pipeline."""

from typing import Any
from pathlib import Path

from config import settings
from core.utils.logger import log


class AgentState:
    def __init__(self, video_path: str, task_description: str = "analyze"):
        self.video_path = video_path
        self.task_description = task_description
        self.transcript = ""
        self.masks: list[dict] = []
        self.manipulation_required = False
        self.output_path = ""
        self.errors: list[str] = []
        self.face_identities: dict[int, Any] = {}


class MediaPipelineAgent:
    def __init__(self):
        self.state: AgentState | None = None
        
    def initialize(self, video_path: str, task: str = "analyze") -> None:
        self.state = AgentState(video_path=video_path, task_description=task)
        log(f"[Agent] Initialized: {Path(video_path).name}, task: {task}")
        
    def run_transcription(self) -> str:
        if self.state is None:
            return ""
        log("[Agent] Transcription...")
        
        from core.processing.transcriber import get_transcriber
        
        try:
            result = get_transcriber().transcribe(Path(self.state.video_path))
            if result:
                self.state.transcript = " ".join([r.get("text", "") for r in result])
                log(f"[Agent] Transcript: {len(self.state.transcript)} chars")
        except Exception as e:
            self.state.errors.append(f"Transcription: {e}")
            log(f"[Agent] Transcription failed: {e}")
        return self.state.transcript
    
    def run_vision(self) -> list[dict]:
        if self.state is None or not settings.enable_sam3_tracking:
            return []
        log("[Agent] SAM3 vision...")
        
        from core.processing.segmentation import Sam3Tracker
        
        try:
            tracker = Sam3Tracker()
            prompts = self._extract_concepts(self.state.task_description)
            for result in tracker.process_video_concepts(Path(self.state.video_path), prompts):
                self.state.masks.append(result)
        except Exception as e:
            self.state.errors.append(f"Vision: {e}")
            log(f"[Agent] Vision failed: {e}")
        return self.state.masks
    
    def run_arbiter(self) -> bool:
        if self.state is None:
            return False
            
        task_lower = self.state.task_description.lower()
        if any(word in task_lower for word in ["remove", "delete", "blur", "hide"]):
            log("[Agent] Manipulation required")
            self.state.manipulation_required = True
            
            if settings.manipulation_backend != "disabled":
                from core.manipulation.arbiter import InpaintingArbiter
                backend = InpaintingArbiter().analyze_scene(Path(self.state.video_path))
                log(f"[Agent] Selected: {backend}")
        return self.state.manipulation_required
    
    def run_manipulation(self) -> str:
        if self.state is None or not self.state.manipulation_required:
            return ""
        if settings.manipulation_backend == "disabled":
            return ""
            
        log("[Agent] Manipulation...")
        from core.manipulation.arbiter import InpaintingArbiter
        from core.manipulation.painters import ProPainterEngine, WanVideoEngine
        
        backend = InpaintingArbiter().analyze_scene(Path(self.state.video_path))
        video_path = Path(self.state.video_path)
        output_path = video_path.parent / f"{video_path.stem}_edited{video_path.suffix}"
        
        engine = ProPainterEngine() if backend == "propainter" else WanVideoEngine()
        if engine.inpaint(video_path, {"masks": self.state.masks}, output_path):
            self.state.output_path = str(output_path)
        return self.state.output_path
    
    def run(self) -> dict[str, Any]:
        if self.state is None:
            return {"error": "Not initialized"}
            
        log("[Agent] Starting pipeline...")
        self.run_transcription()
        self.run_vision()
        self.run_arbiter()
        if self.state.manipulation_required:
            self.run_manipulation()
        log("[Agent] Complete")
        
        return {
            "video_path": self.state.video_path,
            "task": self.state.task_description,
            "transcript_length": len(self.state.transcript),
            "masks_count": len(self.state.masks),
            "manipulation_done": bool(self.state.output_path),
            "output_path": self.state.output_path,
            "errors": self.state.errors,
        }
        
    def _extract_concepts(self, task: str) -> list[str]:
        concepts = []
        task_lower = task.lower()
        if "person" in task_lower or "people" in task_lower:
            concepts.append("person")
        if "car" in task_lower or "vehicle" in task_lower:
            concepts.append("car")
        if "face" in task_lower:
            concepts.append("face")
        return concepts or ["person"]
