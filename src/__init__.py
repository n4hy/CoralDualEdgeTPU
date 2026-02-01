"""Dual Coral Edge TPU interface package."""
from .dual_tpu import (
    DualEdgeTPU, TPUDevice, BBox, DetectionResult, ClassResult,
    list_edge_tpus, make_interpreter, check_tpu_status
)
from .camera import AxisCamera, EmpireTechPTZ, CameraConfig, MultiCameraManager, Frame
from .tracker import (
    BoundingBox, Track, CentroidTracker, IoUTracker,
    detections_to_boxes
)
from .pipeline import DualTPUPipeline, LivePipeline, Detection, FrameResult
from .benchmark import DualTPUBenchmark, BenchmarkResult, ThermalMonitor
from .output import (
    EventPublisher, ObjectFilter, MQTTOutput, WebhookOutput,
    DetectionEvent, SatelliteDetector
)

__all__ = [
    # TPU
    "DualEdgeTPU", "TPUDevice",
    # Camera
    "AxisCamera", "CameraConfig", "MultiCameraManager", "Frame",
    # Tracking
    "BoundingBox", "Track", "CentroidTracker", "IoUTracker",
    "detections_to_boxes",
    # Pipeline
    "DualTPUPipeline", "LivePipeline", "Detection", "FrameResult",
    # Benchmark
    "DualTPUBenchmark", "BenchmarkResult", "ThermalMonitor",
    # Output
    "EventPublisher", "ObjectFilter", "MQTTOutput", "WebhookOutput",
    "DetectionEvent", "SatelliteDetector",
]
