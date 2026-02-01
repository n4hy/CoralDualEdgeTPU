"""Object tracking for detected objects across frames.

Implements:
- Simple centroid tracking
- IoU-based tracking
- Track management (creation, update, deletion)
"""

import time
from dataclasses import dataclass, field
from typing import Optional
from collections import deque

import numpy as np


@dataclass
class BoundingBox:
    """A bounding box with optional class info."""
    x1: float
    y1: float
    x2: float
    y2: float
    class_id: int = -1
    class_name: str = ""
    confidence: float = 0.0

    @property
    def width(self) -> float:
        return self.x2 - self.x1

    @property
    def height(self) -> float:
        return self.y2 - self.y1

    @property
    def center(self) -> tuple[float, float]:
        return ((self.x1 + self.x2) / 2, (self.y1 + self.y2) / 2)

    @property
    def area(self) -> float:
        return self.width * self.height

    def iou(self, other: "BoundingBox") -> float:
        """Calculate Intersection over Union with another box."""
        x1 = max(self.x1, other.x1)
        y1 = max(self.y1, other.y1)
        x2 = min(self.x2, other.x2)
        y2 = min(self.y2, other.y2)

        if x2 <= x1 or y2 <= y1:
            return 0.0

        intersection = (x2 - x1) * (y2 - y1)
        union = self.area + other.area - intersection

        return intersection / union if union > 0 else 0.0


@dataclass
class Track:
    """A tracked object with history."""
    track_id: int
    bbox: BoundingBox
    class_id: int = -1
    class_name: str = ""
    confidence: float = 0.0

    # Track state
    age: int = 0  # Frames since creation
    hits: int = 1  # Successful detections
    misses: int = 0  # Consecutive frames without detection

    # History
    history: deque = field(default_factory=lambda: deque(maxlen=30))

    # Timestamps
    created_at: float = field(default_factory=time.time)
    updated_at: float = field(default_factory=time.time)

    def update(self, bbox: BoundingBox):
        """Update track with new detection."""
        self.history.append(self.bbox)
        self.bbox = bbox
        self.class_id = bbox.class_id
        self.class_name = bbox.class_name
        self.confidence = bbox.confidence
        self.hits += 1
        self.misses = 0
        self.age += 1
        self.updated_at = time.time()

    def mark_missed(self):
        """Mark track as not detected this frame."""
        self.misses += 1
        self.age += 1

    @property
    def velocity(self) -> tuple[float, float]:
        """Estimate velocity from history."""
        if len(self.history) < 2:
            return (0.0, 0.0)
        prev = self.history[-1]
        curr = self.bbox
        return (curr.center[0] - prev.center[0],
                curr.center[1] - prev.center[1])

    def predict_next(self) -> tuple[float, float]:
        """Predict next center position."""
        vx, vy = self.velocity
        cx, cy = self.bbox.center
        return (cx + vx, cy + vy)


class CentroidTracker:
    """Simple centroid-based object tracker.

    Matches detections to existing tracks based on distance
    between centroids. Good for sparse, non-overlapping objects.
    """

    def __init__(self, max_distance: float = 50.0, max_missing: int = 10):
        """
        Args:
            max_distance: Maximum centroid distance for matching
            max_missing: Frames before dropping a track
        """
        self.max_distance = max_distance
        self.max_missing = max_missing
        self.tracks: dict[int, Track] = {}
        self._next_id = 0

    def update(self, detections: list[BoundingBox]) -> list[Track]:
        """Update tracks with new detections.

        Args:
            detections: List of detected bounding boxes

        Returns:
            List of active tracks
        """
        if not detections:
            # Mark all tracks as missed
            for track in self.tracks.values():
                track.mark_missed()
            self._cleanup()
            return list(self.tracks.values())

        if not self.tracks:
            # First frame - create tracks for all detections
            for det in detections:
                self._create_track(det)
            return list(self.tracks.values())

        # Calculate distances between all track centers and detection centers
        track_ids = list(self.tracks.keys())
        track_centers = np.array([self.tracks[tid].bbox.center for tid in track_ids])
        det_centers = np.array([d.center for d in detections])

        # Distance matrix
        distances = np.linalg.norm(
            track_centers[:, np.newaxis] - det_centers[np.newaxis, :],
            axis=2
        )

        # Greedy matching
        matched_tracks = set()
        matched_dets = set()

        while True:
            if distances.size == 0:
                break
            min_idx = np.unravel_index(np.argmin(distances), distances.shape)
            min_dist = distances[min_idx]

            if min_dist > self.max_distance:
                break

            track_idx, det_idx = min_idx
            track_id = track_ids[track_idx]

            if track_id not in matched_tracks and det_idx not in matched_dets:
                self.tracks[track_id].update(detections[det_idx])
                matched_tracks.add(track_id)
                matched_dets.add(det_idx)

            # Mark as processed
            distances[track_idx, :] = np.inf
            distances[:, det_idx] = np.inf

        # Mark unmatched tracks as missed
        for track_id in self.tracks:
            if track_id not in matched_tracks:
                self.tracks[track_id].mark_missed()

        # Create new tracks for unmatched detections
        for i, det in enumerate(detections):
            if i not in matched_dets:
                self._create_track(det)

        self._cleanup()
        return list(self.tracks.values())

    def _create_track(self, bbox: BoundingBox) -> Track:
        """Create a new track."""
        track = Track(
            track_id=self._next_id,
            bbox=bbox,
            class_id=bbox.class_id,
            class_name=bbox.class_name,
            confidence=bbox.confidence
        )
        self.tracks[self._next_id] = track
        self._next_id += 1
        return track

    def _cleanup(self):
        """Remove stale tracks."""
        to_remove = [
            tid for tid, track in self.tracks.items()
            if track.misses > self.max_missing
        ]
        for tid in to_remove:
            del self.tracks[tid]


class IoUTracker:
    """IoU-based object tracker.

    Matches detections to tracks based on bounding box overlap.
    Better for crowded scenes with overlapping objects.
    """

    def __init__(self, iou_threshold: float = 0.3, max_missing: int = 10,
                 min_hits: int = 3):
        """
        Args:
            iou_threshold: Minimum IoU for matching
            max_missing: Frames before dropping a track
            min_hits: Minimum hits before track is confirmed
        """
        self.iou_threshold = iou_threshold
        self.max_missing = max_missing
        self.min_hits = min_hits
        self.tracks: dict[int, Track] = {}
        self._next_id = 0

    def update(self, detections: list[BoundingBox]) -> list[Track]:
        """Update tracks with new detections."""
        if not detections:
            for track in self.tracks.values():
                track.mark_missed()
            self._cleanup()
            return self._get_confirmed_tracks()

        if not self.tracks:
            for det in detections:
                self._create_track(det)
            return self._get_confirmed_tracks()

        # Calculate IoU matrix
        track_ids = list(self.tracks.keys())
        iou_matrix = np.zeros((len(track_ids), len(detections)))

        for i, tid in enumerate(track_ids):
            for j, det in enumerate(detections):
                iou_matrix[i, j] = self.tracks[tid].bbox.iou(det)

        # Hungarian-style matching (greedy for simplicity)
        matched_tracks = set()
        matched_dets = set()

        while True:
            if iou_matrix.size == 0 or np.max(iou_matrix) < self.iou_threshold:
                break

            max_idx = np.unravel_index(np.argmax(iou_matrix), iou_matrix.shape)
            track_idx, det_idx = max_idx
            track_id = track_ids[track_idx]

            self.tracks[track_id].update(detections[det_idx])
            matched_tracks.add(track_id)
            matched_dets.add(det_idx)

            iou_matrix[track_idx, :] = 0
            iou_matrix[:, det_idx] = 0

        # Handle unmatched
        for track_id in self.tracks:
            if track_id not in matched_tracks:
                self.tracks[track_id].mark_missed()

        for i, det in enumerate(detections):
            if i not in matched_dets:
                self._create_track(det)

        self._cleanup()
        return self._get_confirmed_tracks()

    def _create_track(self, bbox: BoundingBox) -> Track:
        track = Track(
            track_id=self._next_id,
            bbox=bbox,
            class_id=bbox.class_id,
            class_name=bbox.class_name,
            confidence=bbox.confidence
        )
        self.tracks[self._next_id] = track
        self._next_id += 1
        return track

    def _cleanup(self):
        to_remove = [
            tid for tid, track in self.tracks.items()
            if track.misses > self.max_missing
        ]
        for tid in to_remove:
            del self.tracks[tid]

    def _get_confirmed_tracks(self) -> list[Track]:
        """Return only confirmed tracks (enough hits)."""
        return [
            track for track in self.tracks.values()
            if track.hits >= self.min_hits or track.misses == 0
        ]


def detections_to_boxes(detections, img_width: int, img_height: int,
                        labels: Optional[dict] = None) -> list[BoundingBox]:
    """Convert pycoral detections to BoundingBox list.

    Args:
        detections: Output from detect.get_objects()
        img_width: Image width for denormalization
        img_height: Image height for denormalization
        labels: Optional class ID to name mapping

    Returns:
        List of BoundingBox objects
    """
    boxes = []
    for det in detections:
        bbox = det.bbox
        box = BoundingBox(
            x1=bbox.xmin * img_width,
            y1=bbox.ymin * img_height,
            x2=bbox.xmax * img_width,
            y2=bbox.ymax * img_height,
            class_id=det.id,
            class_name=labels.get(det.id, str(det.id)) if labels else str(det.id),
            confidence=det.score
        )
        boxes.append(box)
    return boxes
