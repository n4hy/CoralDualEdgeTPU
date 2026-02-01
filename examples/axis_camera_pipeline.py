#!/usr/bin/env python3
"""AXIS camera pipeline example with dual TPU inference.

This example demonstrates:
- Connecting to AXIS M3057-PLVE MK II panoramic cameras
- Running detection on TPU 0
- Running classification on TPU 1 (optional)
- Object tracking across frames
- Live visualization with OpenCV

Before running:
1. Fix TPU permissions (run once):
   echo 'SUBSYSTEM=="apex", MODE="0660", GROUP="plugdev"' | sudo tee /etc/udev/rules.d/99-apex.rules
   sudo udevadm control --reload-rules && sudo udevadm trigger

2. Download models to models/ directory:
   cd models/
   # Detection model (SSD MobileNet)
   wget https://github.com/google-coral/test_data/raw/master/ssd_mobilenet_v2_coco_quant_postprocess_edgetpu.tflite
   wget https://raw.githubusercontent.com/google-coral/test_data/master/coco_labels.txt

   # Classification model (optional)
   wget https://github.com/google-coral/test_data/raw/master/mobilenet_v2_1.0_224_quant_edgetpu.tflite
   wget https://raw.githubusercontent.com/google-coral/test_data/master/imagenet_labels.txt

3. Set your camera IP/credentials below
"""

import sys
import time
from pathlib import Path

import cv2

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src import (
    AxisCamera, CameraConfig, MultiCameraManager,
    DualTPUPipeline, LivePipeline
)


# =============================================================================
# CONFIGURATION - Update these for your setup
# =============================================================================

# Camera settings
CAMERAS = [
    {
        "name": "axis-cam1",
        "ip": "192.168.1.100",  # Update with your camera IP
        "username": "root",     # Update with your credentials
        "password": "password",
        "view": AxisCamera.VIEW_PANORAMIC,  # Or VIEW_QUAD for 4-up view
        "resolution": "1280x720",
    },
    # Add second camera when available:
    # {
    #     "name": "axis-cam2",
    #     "ip": "192.168.1.101",
    #     "username": "root",
    #     "password": "password",
    #     "view": AxisCamera.VIEW_PANORAMIC,
    #     "resolution": "1280x720",
    # },
]

# Model paths (relative to project root)
PROJECT_ROOT = Path(__file__).parent.parent
DETECTION_MODEL = PROJECT_ROOT / "models" / "ssd_mobilenet_v2_coco_quant_postprocess_edgetpu.tflite"
DETECTION_LABELS = PROJECT_ROOT / "models" / "coco_labels.txt"
CLASSIFICATION_MODEL = PROJECT_ROOT / "models" / "mobilenet_v2_1.0_224_quant_edgetpu.tflite"
CLASSIFICATION_LABELS = PROJECT_ROOT / "models" / "imagenet_labels.txt"

# Detection settings
DETECTION_THRESHOLD = 0.5
CLASSIFICATION_THRESHOLD = 0.3

# Display settings
SHOW_DISPLAY = True
DISPLAY_WIDTH = 1280
DISPLAY_HEIGHT = 720


def print_result(result):
    """Print detection results to console."""
    if result.detections:
        det_str = ", ".join(
            f"#{d.track_id}:{d.class_name}({d.confidence:.2f})"
            for d in result.detections
        )
        print(f"[{result.camera_name}] Frame {result.frame_number}: "
              f"{len(result.detections)} objects - {det_str} "
              f"({result.processing_time_ms:.1f}ms)")


def main():
    # Check for models
    if not DETECTION_MODEL.exists():
        print(f"Detection model not found: {DETECTION_MODEL}")
        print("\nDownload models first:")
        print("  cd models/")
        print("  wget https://github.com/google-coral/test_data/raw/master/"
              "ssd_mobilenet_v2_coco_quant_postprocess_edgetpu.tflite")
        print("  wget https://raw.githubusercontent.com/google-coral/test_data/"
              "master/coco_labels.txt")
        return 1

    # Initialize pipeline
    print("Initializing dual TPU pipeline...")
    try:
        pipeline = DualTPUPipeline(
            detection_model=str(DETECTION_MODEL),
            classification_model=str(CLASSIFICATION_MODEL) if CLASSIFICATION_MODEL.exists() else None,
            detection_threshold=DETECTION_THRESHOLD,
            classification_threshold=CLASSIFICATION_THRESHOLD,
            labels_path=str(DETECTION_LABELS) if DETECTION_LABELS.exists() else None,
            class_labels_path=str(CLASSIFICATION_LABELS) if CLASSIFICATION_LABELS.exists() else None,
            tracker_type="iou"
        )
    except Exception as e:
        print(f"Failed to initialize pipeline: {e}")
        return 1

    # Create live pipeline
    live = LivePipeline(pipeline)

    # Add cameras
    for cam_cfg in CAMERAS:
        print(f"Adding camera: {cam_cfg['name']} at {cam_cfg['ip']}")
        live.add_axis_camera(
            name=cam_cfg["name"],
            ip=cam_cfg["ip"],
            username=cam_cfg.get("username", ""),
            password=cam_cfg.get("password", ""),
            view=cam_cfg.get("view", ""),
            resolution=cam_cfg.get("resolution", "1280x720")
        )

    # Register callback
    live.on_result(print_result)

    # Start processing
    print("\nStarting live pipeline...")
    print("Press 'q' to quit, 's' to save screenshot")
    print("-" * 60)

    with live:
        if SHOW_DISPLAY:
            cv2.namedWindow("Dual TPU Pipeline", cv2.WINDOW_NORMAL)
            cv2.resizeWindow("Dual TPU Pipeline", DISPLAY_WIDTH, DISPLAY_HEIGHT)

        try:
            while True:
                result = live.get_result(timeout=0.1)

                if result and result.image is not None and SHOW_DISPLAY:
                    # Add stats overlay
                    stats_text = (
                        f"FPS: {1000/result.processing_time_ms:.1f} | "
                        f"Objects: {len(result.detections)} | "
                        f"Camera: {result.camera_name}"
                    )
                    cv2.putText(
                        result.image, stats_text, (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2
                    )

                    cv2.imshow("Dual TPU Pipeline", result.image)

                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    break
                elif key == ord('s') and result and result.image is not None:
                    filename = f"screenshot_{int(time.time())}.jpg"
                    cv2.imwrite(filename, result.image)
                    print(f"Saved: {filename}")

        except KeyboardInterrupt:
            print("\nInterrupted by user")

    if SHOW_DISPLAY:
        cv2.destroyAllWindows()

    # Print final stats
    print("\n" + "=" * 60)
    print("Session Statistics:")
    print(f"  Frames processed: {pipeline.stats['frames_processed']}")
    print(f"  Total detections: {pipeline.stats['total_detections']}")
    print(f"  Avg inference time: {pipeline.stats['avg_inference_ms']:.1f}ms")

    return 0


if __name__ == "__main__":
    sys.exit(main())
