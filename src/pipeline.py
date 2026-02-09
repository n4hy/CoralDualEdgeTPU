"""Inference pipeline combining cameras, TPUs, and tracking.

Supports:
- Multi-camera input
- Dual TPU inference (detection + classification)
- Object tracking across frames
- Configurable processing strategies
"""

import threading
import time
from dataclasses import dataclass
from queue import Queue, Empty
from typing import Optional, Callable

import cv2
import numpy as np

from .dual_tpu import DualEdgeTPU
from .camera import Frame, MultiCameraManager, CameraConfig
from .tracker import (
    IoUTracker, CentroidTracker, BoundingBox,
    detections_to_boxes
)


@dataclass
class Detection:
    """A detection result with tracking info."""
    track_id: int
    bbox: BoundingBox
    class_id: int
    class_name: str
    confidence: float
    classification: Optional[str] = None
    classification_confidence: float = 0.0


@dataclass
class FrameResult:
    """Result of processing a single frame."""
    camera_name: str
    frame_number: int
    timestamp: float
    processing_time_ms: float
    detections: list[Detection]
    image: Optional[np.ndarray] = None  # Annotated image if requested


class DualTPUPipeline:
    """Video analytics pipeline using dual Edge TPUs.

    Architecture:
    - TPU 0: Object detection (SSD MobileNet, EfficientDet, etc.)
    - TPU 1: Classification of detected objects

    This allows detection and classification to run in parallel
    on different objects or frames.
    """

    def __init__(self,
                 detection_model: str,
                 classification_model: Optional[str] = None,
                 detection_threshold: float = 0.5,
                 classification_threshold: float = 0.3,
                 labels_path: Optional[str] = None,
                 class_labels_path: Optional[str] = None,
                 tracker_type: str = "iou"):
        """
        Args:
            detection_model: Path to Edge TPU detection model
            classification_model: Path to Edge TPU classification model (optional)
            detection_threshold: Minimum detection confidence
            classification_threshold: Minimum classification confidence
            labels_path: Path to detection labels file
            class_labels_path: Path to classification labels file
            tracker_type: "iou" or "centroid"
        """
        self.detection_model = detection_model
        self.classification_model = classification_model
        self.detection_threshold = detection_threshold
        self.classification_threshold = classification_threshold

        # Load labels
        self.det_labels = self._load_labels(labels_path) if labels_path else {}
        self.class_labels = self._load_labels(class_labels_path) if class_labels_path else {}

        # Initialize TPUs
        self.tpu = DualEdgeTPU()

        # Load models
        print(f"Loading detection model on TPU 0...")
        self.tpu.load_model(detection_model, device_idx=0)

        if classification_model and self.tpu.num_devices > 1:
            print(f"Loading classification model on TPU 1...")
            self.tpu.load_model(classification_model, device_idx=1)
            self._dual_mode = True
        else:
            self._dual_mode = False

        # Get model input sizes
        self.det_input_size = self._get_input_size(0)
        if self._dual_mode:
            self.class_input_size = self._get_input_size(1)

        # Trackers per camera
        self.trackers: dict[str, IoUTracker | CentroidTracker] = {}
        self.tracker_type = tracker_type
        self._lock = threading.Lock()

        # Processing stats
        self.stats = {
            "frames_processed": 0,
            "total_detections": 0,
            "avg_inference_ms": 0.0
        }

    def _load_labels(self, path: str) -> dict[int, str]:
        """Load labels file (one label per line)."""
        labels = {}
        with open(path, "r") as f:
            for i, line in enumerate(f):
                labels[i] = line.strip()
        return labels

    def _get_input_size(self, device_idx: int) -> tuple[int, int]:
        """Get model input size (height, width)."""
        details = self.tpu.get_input_details(device_idx)
        shape = details["shape"]
        # Typically [1, height, width, 3]
        return (shape[1], shape[2])

    def _get_tracker(self, camera_name: str):
        """Get or create tracker for a camera."""
        with self._lock:
            if camera_name not in self.trackers:
                if self.tracker_type == "iou":
                    self.trackers[camera_name] = IoUTracker(
                        iou_threshold=0.3,
                        max_missing=15
                    )
                else:
                    self.trackers[camera_name] = CentroidTracker(
                        max_distance=50.0,
                        max_missing=15
                    )
            return self.trackers[camera_name]

    def preprocess(self, image: np.ndarray, target_size: tuple[int, int]) -> np.ndarray:
        """Preprocess image for inference."""
        # Resize
        resized = cv2.resize(image, (target_size[1], target_size[0]))
        # BGR to RGB
        rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
        return rgb

    def process_frame(self, frame: Frame,
                      annotate: bool = False) -> FrameResult:
        """Process a single frame through the pipeline.

        Args:
            frame: Input frame from camera
            annotate: Whether to draw detections on image

        Returns:
            FrameResult with detections and tracks
        """
        start_time = time.time()

        # Preprocess for detection
        det_input = self.preprocess(frame.image, self.det_input_size)

        # Run detection on TPU 0
        raw_detections = self.tpu.detect(
            det_input,
            device_idx=0,
            threshold=self.detection_threshold
        )

        # Convert to bounding boxes
        h, w = frame.image.shape[:2]
        boxes = detections_to_boxes(raw_detections, w, h, self.det_labels)

        # Update tracker
        tracker = self._get_tracker(frame.camera_name)
        tracks = tracker.update(boxes)

        # Build detection results
        detections = []
        for track in tracks:
            det = Detection(
                track_id=track.track_id,
                bbox=track.bbox,
                class_id=track.class_id,
                class_name=track.class_name,
                confidence=track.confidence
            )

            # Run classification on TPU 1 if available
            if self._dual_mode and track.misses == 0:
                cls_result = self._classify_crop(frame.image, track.bbox)
                if cls_result:
                    det.classification = cls_result[0]
                    det.classification_confidence = cls_result[1]

            detections.append(det)

        processing_time = (time.time() - start_time) * 1000

        # Update stats
        with self._lock:
            self.stats["frames_processed"] += 1
            self.stats["total_detections"] += len(detections)
            alpha = 0.1
            self.stats["avg_inference_ms"] = (
                alpha * processing_time +
                (1 - alpha) * self.stats["avg_inference_ms"]
            )

        # Annotate if requested
        annotated = None
        if annotate:
            annotated = self._annotate_frame(frame.image.copy(), detections)

        return FrameResult(
            camera_name=frame.camera_name,
            frame_number=frame.frame_number,
            timestamp=frame.timestamp,
            processing_time_ms=processing_time,
            detections=detections,
            image=annotated
        )

    def _classify_crop(self, image: np.ndarray,
                       bbox: BoundingBox) -> Optional[tuple[str, float]]:
        """Crop and classify a detected region."""
        # Extract crop
        x1, y1 = int(bbox.x1), int(bbox.y1)
        x2, y2 = int(bbox.x2), int(bbox.y2)

        # Ensure valid crop
        h, w = image.shape[:2]
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(w, x2), min(h, y2)

        if x2 <= x1 or y2 <= y1:
            return None

        crop = image[y1:y2, x1:x2]

        # Preprocess for classification
        cls_input = self.preprocess(crop, self.class_input_size)

        # Run classification on TPU 1
        results = self.tpu.classify(
            cls_input,
            device_idx=1,
            top_k=1,
            threshold=self.classification_threshold
        )

        if results:
            class_id = results[0].id
            score = results[0].score
            class_name = self.class_labels.get(class_id, str(class_id))
            return (class_name, score)

        return None

    def _annotate_frame(self, image: np.ndarray,
                        detections: list[Detection]) -> np.ndarray:
        """Draw detections on image."""
        for det in detections:
            bbox = det.bbox
            x1, y1 = int(bbox.x1), int(bbox.y1)
            x2, y2 = int(bbox.x2), int(bbox.y2)

            # Color by track ID
            color = self._id_to_color(det.track_id)

            # Draw box
            cv2.rectangle(image, (x1, y1), (x2, y2), color, 2)

            # Label
            label = f"#{det.track_id} {det.class_name}"
            if det.classification:
                label += f" [{det.classification}]"

            # Background for text
            (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
            cv2.rectangle(image, (x1, y1 - th - 4), (x1 + tw, y1), color, -1)
            cv2.putText(image, label, (x1, y1 - 2),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

        return image

    def _id_to_color(self, track_id: int) -> tuple[int, int, int]:
        """Generate consistent color for track ID."""
        h = hash(track_id * 42) & 0xFFFFFF
        return (50 + (h & 0xFF) % 206, 50 + ((h >> 8) & 0xFF) % 206, 50 + ((h >> 16) & 0xFF) % 206)


class LivePipeline:
    """Live video processing pipeline with multi-camera support."""

    def __init__(self, pipeline: DualTPUPipeline):
        self.pipeline = pipeline
        self.cameras = MultiCameraManager()
        self._running = False
        self._results_queue: Queue[FrameResult] = Queue(maxsize=100)
        self._threads: list[threading.Thread] = []
        self._callbacks: list[Callable[[FrameResult], None]] = []

    def add_camera(self, config: CameraConfig):
        """Add a camera to the pipeline."""
        self.cameras.add_camera(config)

    def add_axis_camera(self, name: str, ip: str,
                        username: str = "", password: str = "",
                        view: str = "", resolution: str = "1280x720"):
        """Add an AXIS camera."""
        self.cameras.add_axis_camera(name, ip, username, password, view, resolution)

    def on_result(self, callback: Callable[[FrameResult], None]):
        """Register callback for processing results."""
        self._callbacks.append(callback)

    def start(self):
        """Start the live pipeline."""
        self._running = True
        self.cameras.start_all()

        # Start processing thread for each camera
        for name, camera in self.cameras.cameras.items():
            t = threading.Thread(
                target=self._process_camera,
                args=(name, camera),
                daemon=True
            )
            self._threads.append(t)
            t.start()

        print(f"Pipeline started with {len(self.cameras.cameras)} camera(s)")

    def stop(self):
        """Stop the pipeline."""
        self._running = False
        self.cameras.stop_all()
        for t in self._threads:
            t.join(timeout=2.0)
        self._threads.clear()

    def _process_camera(self, name: str, camera):
        """Process frames from a single camera."""
        while self._running:
            frame = camera.get_frame(timeout=0.5)
            if frame is None:
                continue

            try:
                result = self.pipeline.process_frame(frame, annotate=True)

                # Queue result
                if not self._results_queue.full():
                    self._results_queue.put(result)

                # Callbacks
                for cb in self._callbacks:
                    try:
                        cb(result)
                    except Exception as e:
                        print(f"Callback error: {e}")

            except Exception as e:
                print(f"Processing error for {name}: {e}")

    def get_result(self, timeout: float = 1.0) -> Optional[FrameResult]:
        """Get next processing result."""
        try:
            return self._results_queue.get(timeout=timeout)
        except Empty:
            return None

    def __enter__(self):
        self.start()
        return self

    def __exit__(self, *args):
        self.stop()
