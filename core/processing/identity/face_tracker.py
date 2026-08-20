"""Temporal Face Grouping for Accurate Clustering."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from config import settings
from core.domain.schemas import DetectedFace

# Face tracking score weights (IoU position overlap vs embedding similarity)
_IOU_WEIGHT = 0.4
_SIMILARITY_WEIGHT = 0.6


@dataclass
class ActiveFaceTrack:
    """A face track being built during video processing.

    Tracks group face detections across consecutive frames based on:
    1. Spatial continuity (IoU overlap of bounding boxes)
    2. Appearance consistency (cosine similarity of embeddings)
    """

    track_id: int
    start_frame: int
    end_frame: int
    start_time: float
    end_time: float
    embeddings: list[list[float]]  # All embeddings in this track
    bboxes: list[tuple[int, int, int, int]]  # All bounding boxes
    confidences: list[float]  # Detection confidences
    best_thumbnail_frame: int = 0  # Frame with highest confidence
    best_confidence: float = 0.0


class FaceTrackBuilder:
    """Build face tracks from frame-by-frame detections.

    The key insight: Instead of clustering all faces globally (which fails
    with lighting/angle variation), we first group faces into temporal TRACKS
    within each video. A track is a sequence of the same face across frames.

    Algorithm:
    1. For each new frame, match detected faces to existing active tracks
    2. Matching uses IoU (spatial overlap) + cosine similarity (appearance)
    3. If no match, start a new track
    4. If a track has no matches for N frames, finalize it
    5. Output: List of finalized tracks with averaged embeddings

    This solves the "Prakash in dark room" vs "Prakash outside" problem
    because the track contains multiple angles/lighting conditions, and
    the averaged embedding is more robust.
    """

    # Thresholds
    IOU_THRESHOLD: float = settings.face_track_iou_threshold
    COSINE_THRESHOLD: float = settings.face_track_cosine_threshold
    MAX_FRAMES_MISSING: int = settings.face_track_max_missing_frames
    MIN_TRACK_LENGTH: int = 2  # Minimum frames to consider a valid track

    def __init__(self, frame_interval: float = 1.0) -> None:
        """Initialize the track builder.

        Args:
            frame_interval: Seconds between frames (for timestamp calculation).
        """
        self.frame_interval = frame_interval
        self.active_tracks: dict[int, ActiveFaceTrack] = {}
        self.finalized_tracks: list[ActiveFaceTrack] = []
        self._next_track_id = 1
        self._current_frame = 0
        self._frames_since_last_seen: dict[int, int] = {}  # track_id -> frames

    def process_frame(
        self,
        faces: list[DetectedFace],
        frame_index: int,
        timestamp: float,
    ) -> None:
        """Process detected faces from a single frame.

        Args:
            faces: List of detected faces with embeddings and bboxes.
            frame_index: Current frame number.
            timestamp: Current timestamp in seconds.
        """
        self._current_frame = frame_index

        # Track which active tracks got matched this frame
        matched_track_ids: set[int] = set()

        for face in faces:
            if face.embedding is None:
                continue

            # Try to match to an existing active track
            best_track_id = self._find_best_matching_track(face)

            if best_track_id is not None:
                # Extend existing track
                track = self.active_tracks[best_track_id]
                track.end_frame = frame_index
                track.end_time = timestamp
                track.embeddings.append(face.embedding)
                track.bboxes.append(face.bbox)
                track.confidences.append(face.confidence)

                # Update best thumbnail if this is higher confidence
                if face.confidence > track.best_confidence:
                    track.best_confidence = face.confidence
                    track.best_thumbnail_frame = frame_index

                matched_track_ids.add(best_track_id)
                self._frames_since_last_seen[best_track_id] = 0
            else:
                # Start new track
                new_track = ActiveFaceTrack(
                    track_id=self._next_track_id,
                    start_frame=frame_index,
                    end_frame=frame_index,
                    start_time=timestamp,
                    end_time=timestamp,
                    embeddings=[face.embedding],
                    bboxes=[face.bbox],
                    confidences=[face.confidence],
                    best_thumbnail_frame=frame_index,
                    best_confidence=face.confidence,
                )
                self.active_tracks[self._next_track_id] = new_track
                self._frames_since_last_seen[self._next_track_id] = 0
                self._next_track_id += 1

        # Update frames since last seen for unmatched tracks
        tracks_to_finalize = []
        for track_id in list(self.active_tracks.keys()):
            if track_id not in matched_track_ids:
                self._frames_since_last_seen[track_id] = (
                    self._frames_since_last_seen.get(track_id, 0) + 1
                )
                if (
                    self._frames_since_last_seen[track_id]
                    >= self.MAX_FRAMES_MISSING
                ):
                    tracks_to_finalize.append(track_id)

        # Finalize tracks that have been missing too long
        for track_id in tracks_to_finalize:
            self._finalize_track(track_id)

    def _find_best_matching_track(self, face: DetectedFace) -> int | None:
        """Find the best matching active track for a face detection.

        Uses both IoU (spatial) and cosine similarity (appearance) matching.

        Returns:
            Track ID if a good match is found, None otherwise.
        """
        if not self.active_tracks:
            return None

        best_track_id = None
        best_score = 0.0

        face_emb = np.array(face.embedding, dtype=np.float64)
        face_emb_norm = face_emb / (np.linalg.norm(face_emb) + 1e-9)

        for track_id, track in self.active_tracks.items():
            # Skip if track hasn't been seen recently (likely a different person)
            if self._frames_since_last_seen.get(track_id, 0) > 2:
                continue

            # Compute IoU with last known bbox
            last_bbox = track.bboxes[-1]
            iou = self._compute_iou(face.bbox, last_bbox)

            if iou < self.IOU_THRESHOLD:
                continue

            # Compute cosine similarity with track centroid
            track_centroid = self._compute_track_centroid(track)
            similarity = float(np.dot(face_emb_norm, track_centroid))

            if similarity < self.COSINE_THRESHOLD:
                continue

            # Combined score: weighted average of IoU and similarity
            combined_score = _IOU_WEIGHT * iou + _SIMILARITY_WEIGHT * similarity

            if combined_score > best_score:
                best_score = combined_score
                best_track_id = track_id

        return best_track_id

    def _compute_iou(
        self,
        box1: tuple[int, int, int, int],
        box2: tuple[int, int, int, int],
    ) -> float:
        """Compute Intersection over Union between two bboxes.

        Boxes are in (top, right, bottom, left) format.
        """
        top1, right1, bottom1, left1 = box1
        top2, right2, bottom2, left2 = box2

        # Intersection
        inter_left = max(left1, left2)
        inter_top = max(top1, top2)
        inter_right = min(right1, right2)
        inter_bottom = min(bottom1, bottom2)

        if inter_right <= inter_left or inter_bottom <= inter_top:
            return 0.0

        inter_area = (inter_right - inter_left) * (inter_bottom - inter_top)

        # Union
        area1 = (right1 - left1) * (bottom1 - top1)
        area2 = (right2 - left2) * (bottom2 - top2)
        union_area = area1 + area2 - inter_area

        if union_area <= 0:
            return 0.0

        return inter_area / union_area

    # Default embedding dimension for zero-vector fallback
    _EMBEDDING_DIM_DEFAULT: int = 512

    def _compute_track_centroid(self, track: ActiveFaceTrack) -> np.ndarray:
        """Compute normalized centroid embedding for a track."""
        if not track.embeddings:
            return np.zeros(self._EMBEDDING_DIM_DEFAULT, dtype=np.float64)

        embeddings = np.array(track.embeddings, dtype=np.float64)
        centroid = np.mean(embeddings, axis=0)
        centroid /= np.linalg.norm(centroid) + 1e-9
        return centroid

    def _finalize_track(self, track_id: int) -> None:
        """Move a track from active to finalized."""
        if track_id not in self.active_tracks:
            return

        track = self.active_tracks.pop(track_id)
        del self._frames_since_last_seen[track_id]

        # Only keep tracks with minimum length
        if len(track.embeddings) >= self.MIN_TRACK_LENGTH:
            self.finalized_tracks.append(track)

    def finalize_all(self) -> list[ActiveFaceTrack]:
        """Finalize all remaining active tracks and return all tracks.

        Call this at the end of video processing.
        """
        for track_id in list(self.active_tracks.keys()):
            self._finalize_track(track_id)

        return self.finalized_tracks

    def get_track_embeddings(self) -> list[tuple[int, list[float], dict]]:
        """Get averaged embeddings for all finalized tracks.

        Returns:
            List of (track_id, averaged_embedding, metadata) tuples.
        """
        results = []
        for track in self.finalized_tracks:
            centroid = self._compute_track_centroid(track)
            metadata = {
                "start_frame": track.start_frame,
                "end_frame": track.end_frame,
                "start_time": track.start_time,
                "end_time": track.end_time,
                "frame_count": len(track.embeddings),
                "best_thumbnail_frame": track.best_thumbnail_frame,
                "avg_confidence": np.mean(track.confidences)
                if track.confidences
                else 0.0,
            }
            results.append((track.track_id, centroid.tolist(), metadata))

        return results
