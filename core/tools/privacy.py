from __future__ import annotations
import tempfile
import subprocess
from pathlib import Path
from dataclasses import dataclass
from typing import Callable

import cv2
import numpy as np

from config import settings
from core.storage.identity_graph import identity_graph
from core.processing.identity import FaceManager
from core.utils.logger import log


@dataclass
class RedactionResult:
    success: bool
    output_path: str
    frames_processed: int
    faces_blurred: int
    error: str = ""


class VideoRedactor:
    def __init__(self, similarity_threshold: float | None = None):
        self.threshold = similarity_threshold or settings.face_clustering_threshold
        self._face_manager: FaceManager | None = None
    
    @property
    def face_manager(self) -> FaceManager:
        if self._face_manager is None:
            self._face_manager = FaceManager(use_gpu=settings.device == "cuda")
        return self._face_manager
    
    async def redact_identity(
        self,
        video_path: Path,
        target_identity_id: str,
        output_path: Path | None = None,
        progress_callback: Callable[[int, int], None] | None = None,
    ) -> RedactionResult:
        video_path = Path(video_path)
        if not video_path.exists():
            return RedactionResult(False, "", 0, 0, "Video not found")
        
        target_embedding = self._get_identity_embedding(target_identity_id)
        if target_embedding is None:
            return RedactionResult(False, "", 0, 0, "Identity not found or no embeddings")
        
        if output_path is None:
            output_path = video_path.parent / f"{video_path.stem}_redacted{video_path.suffix}"
        output_path = Path(output_path)
        
        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            return RedactionResult(False, "", 0, 0, "Cannot open video")
        
        fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        temp_video = tempfile.NamedTemporaryFile(suffix='.mp4', delete=False)
        writer = cv2.VideoWriter(temp_video.name, fourcc, fps, (width, height))
        
        frames_processed = 0
        faces_blurred = 0
        process_every_n = max(1, int(fps // 5))  # Process 5 fps for detection
        last_boxes: list[tuple[int, int, int, int]] = []
        
        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                
                if frames_processed % process_every_n == 0:
                    boxes = await self._detect_and_match(frame, target_embedding)
                    last_boxes = boxes
                
                for box in last_boxes:
                    frame = self._apply_blur(frame, box)
                    faces_blurred += 1
                
                writer.write(frame)
                frames_processed += 1
                
                if progress_callback and frames_processed % 30 == 0:
                    progress_callback(frames_processed, total_frames)
        finally:
            cap.release()
            writer.release()
        
        success = self._mux_audio(video_path, Path(temp_video.name), output_path)
        
        try:
            Path(temp_video.name).unlink()
        except Exception:
            pass
        
        if success:
            return RedactionResult(True, str(output_path), frames_processed, faces_blurred)
        return RedactionResult(False, "", frames_processed, faces_blurred, "Audio mux failed")
    
    def _get_identity_embedding(self, identity_id: str) -> np.ndarray | None:
        identity = identity_graph.get_identity(identity_id)
        if not identity:
            return None
        
        tracks = identity_graph.get_tracks_for_identity(identity_id)
        if not tracks:
            return None
        
        embeddings = []
        for track in tracks:
            if track.avg_embedding is not None:
                embeddings.append(np.array(track.avg_embedding, dtype=np.float64))
        
        if not embeddings:
            return None
        
        avg = np.mean(embeddings, axis=0)
        avg /= np.linalg.norm(avg) + 1e-9
        return avg
    
    async def _detect_and_match(
        self,
        frame: np.ndarray,
        target_embedding: np.ndarray,
    ) -> list[tuple[int, int, int, int]]:
        import tempfile
        temp_path = tempfile.NamedTemporaryFile(suffix='.jpg', delete=False)
        cv2.imwrite(temp_path.name, frame)
        
        try:
            faces = await self.face_manager.detect_faces(temp_path.name)
        except Exception:
            return []
        finally:
            try:
                Path(temp_path.name).unlink()
            except Exception:
                pass
        
        matched_boxes = []
        for face in faces:
            if face.embedding is None:
                continue
            
            face_emb = np.array(face.embedding, dtype=np.float64)
            face_emb /= np.linalg.norm(face_emb) + 1e-9
            
            similarity = float(np.dot(face_emb, target_embedding))
            if similarity > (1.0 - self.threshold):
                matched_boxes.append(face.bbox)
        
        return matched_boxes
    
    def _apply_blur(self, frame: np.ndarray, box: tuple[int, int, int, int]) -> np.ndarray:
        top, right, bottom, left = box
        h, w = frame.shape[:2]
        
        pad = int((bottom - top) * 0.1)
        y1 = max(0, top - pad)
        y2 = min(h, bottom + pad)
        x1 = max(0, left - pad)
        x2 = min(w, right + pad)
        
        if y2 > y1 and x2 > x1:
            roi = frame[y1:y2, x1:x2]
            blur_size = max(51, ((y2 - y1) // 3) | 1)
            blurred = cv2.GaussianBlur(roi, (blur_size, blur_size), 30)
            frame[y1:y2, x1:x2] = blurred
        
        return frame
    
    def _mux_audio(self, original: Path, video_only: Path, output: Path) -> bool:
        cmd = [
            "ffmpeg", "-y",
            "-i", str(video_only),
            "-i", str(original),
            "-c:v", "copy",
            "-c:a", "aac",
            "-map", "0:v:0",
            "-map", "1:a:0?",
            "-shortest",
            str(output),
        ]
        try:
            result = subprocess.run(cmd, capture_output=True, timeout=300)
            return result.returncode == 0
        except Exception:
            return False
