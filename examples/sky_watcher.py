#!/usr/bin/env python3
"""Sky watcher pipeline for airplane and satellite detection.

Uses dual TPUs to detect and track:
- Airplanes (daytime and nighttime with lights)
- Satellites (solar-illuminated at night)

Publishes detections via MQTT and/or webhooks.

Configuration:
- Edit CAMERAS section for your AXIS camera IPs
- Edit MQTT_CONFIG for your broker
- Edit WEBHOOK_CONFIG for your endpoint
"""

import sys
import time
from datetime import datetime
from pathlib import Path

import cv2

sys.path.insert(0, str(Path(__file__).parent.parent))

from src import (
    AxisCamera, EmpireTechPTZ, CameraConfig, MultiCameraManager,
    DualTPUPipeline, LivePipeline
)
from src.output import (
    EventPublisher, ObjectFilter, MQTTOutput, WebhookOutput,
    DetectionEvent, SatelliteDetector
)


# =============================================================================
# CONFIGURATION
# =============================================================================

# Cameras - update with your camera settings
# Camera 1: AXIS M3057-PLVE MK II (panoramic dome)
# Camera 2: Empire Tech PTZ425DB-AT (4MP 25x PTZ)

CAMERAS = [
    {
        "type": "axis",
        "name": "axis-panoramic",
        "ip": "192.168.1.100",  # UPDATE THIS
        "username": "root",
        "password": "password",  # UPDATE THIS
        "view": AxisCamera.VIEW_PANORAMIC,  # Full 360 view for sky watching
        "resolution": "1920x1080",
    },
    {
        "type": "empiretech",
        "name": "ptz-tracker",
        "ip": "192.168.1.101",  # UPDATE THIS
        "username": "admin",
        "password": "password",  # UPDATE THIS
        "main_stream": True,  # 4MP full resolution
    },
]

# MQTT Configuration
MQTT_CONFIG = {
    "enabled": True,
    "broker": "localhost",  # UPDATE: your MQTT broker
    "port": 1883,
    "username": None,       # Set if broker requires auth
    "password": None,
    "topic_prefix": "sky-watcher/detections",
}

# Webhook Configuration
WEBHOOK_CONFIG = {
    "enabled": False,
    "url": "http://localhost:8080/api/detections",  # UPDATE: your endpoint
    "headers": {
        "Content-Type": "application/json",
        "Authorization": "Bearer YOUR_TOKEN",  # If needed
    },
}

# Models
PROJECT_ROOT = Path(__file__).parent.parent
DETECTION_MODEL = PROJECT_ROOT / "models" / "ssd_mobilenet_v2_coco_quant_postprocess_edgetpu.tflite"
DETECTION_LABELS = PROJECT_ROOT / "models" / "coco_labels.txt"

# For better airplane detection, consider:
# - EfficientDet for higher accuracy
# - Custom-trained model for aircraft

# Detection settings
DETECTION_THRESHOLD = 0.4  # Lower for distant aircraft
CLASSIFICATION_THRESHOLD = 0.3

# Classes to detect (COCO dataset)
# 4 = airplane
INCLUDE_CLASSES = {4, "airplane", "aeroplane", "bird", "kite"}


# =============================================================================
# MAIN
# =============================================================================

def log_detection(event: DetectionEvent):
    """Log detection to console."""
    timestamp = datetime.now().strftime("%H:%M:%S")
    emoji = ""
    if "airplane" in event.class_name.lower():
        emoji = "[AIRCRAFT]"
    elif "satellite" in event.class_name.lower():
        emoji = "[SATELLITE]"

    print(f"[{timestamp}] {emoji} {event.event_type.upper()}: "
          f"Track #{event.track_id} - {event.class_name} "
          f"(conf: {event.confidence:.2f}) @ {event.camera_name}")


def main():
    print("="*60)
    print("SKY WATCHER - Airplane & Satellite Detection")
    print("="*60)

    # Check for model
    if not DETECTION_MODEL.exists():
        print(f"\nModel not found: {DETECTION_MODEL}")
        print("\nDownload with:")
        print("  cd models/")
        print("  wget https://github.com/google-coral/test_data/raw/master/"
              "ssd_mobilenet_v2_coco_quant_postprocess_edgetpu.tflite")
        print("  wget https://raw.githubusercontent.com/google-coral/test_data/"
              "master/coco_labels.txt")
        return 1

    # Initialize dual TPU pipeline
    print("\nInitializing dual TPU pipeline...")
    try:
        pipeline = DualTPUPipeline(
            detection_model=str(DETECTION_MODEL),
            classification_model=None,  # Detection only for sky watching
            detection_threshold=DETECTION_THRESHOLD,
            labels_path=str(DETECTION_LABELS) if DETECTION_LABELS.exists() else None,
            tracker_type="iou"
        )
    except Exception as e:
        print(f"Failed to initialize pipeline: {e}")
        print("\nCheck TPU permissions:")
        print("  ls -la /dev/apex*")
        return 1

    # Setup event publishing
    print("\nSetting up event publishing...")
    object_filter = ObjectFilter(
        include_classes=INCLUDE_CLASSES,
        min_confidence=DETECTION_THRESHOLD,
        airplane_mode=True,
        satellite_mode=True
    )

    publisher = EventPublisher(
        object_filter=object_filter,
        rate_limit_per_track=0.5  # Max 2 events/sec per track
    )

    # Add MQTT
    if MQTT_CONFIG["enabled"]:
        print(f"  Connecting to MQTT broker: {MQTT_CONFIG['broker']}...")
        try:
            publisher.add_mqtt(
                broker=MQTT_CONFIG["broker"],
                port=MQTT_CONFIG["port"],
                username=MQTT_CONFIG.get("username"),
                password=MQTT_CONFIG.get("password"),
                topic_prefix=MQTT_CONFIG["topic_prefix"]
            )
            print(f"  MQTT connected - publishing to {MQTT_CONFIG['topic_prefix']}/*")
        except Exception as e:
            print(f"  MQTT connection failed: {e}")

    # Add webhook
    if WEBHOOK_CONFIG["enabled"]:
        print(f"  Adding webhook: {WEBHOOK_CONFIG['url']}")
        publisher.add_webhook(
            url=WEBHOOK_CONFIG["url"],
            headers=WEBHOOK_CONFIG.get("headers")
        )

    # Create live pipeline
    live = LivePipeline(pipeline)

    # Add cameras
    print("\nConfiguring cameras...")
    for cam in CAMERAS:
        cam_type = cam.get("type", "axis")
        print(f"  Adding {cam_type}: {cam['name']} at {cam['ip']}")

        if cam_type == "empiretech":
            live.cameras.add_empiretech_ptz(
                name=cam["name"],
                ip=cam["ip"],
                username=cam.get("username", ""),
                password=cam.get("password", ""),
                main_stream=cam.get("main_stream", True)
            )
        else:  # axis or default
            live.add_axis_camera(
                name=cam["name"],
                ip=cam["ip"],
                username=cam.get("username", ""),
                password=cam.get("password", ""),
                view=cam.get("view", ""),
                resolution=cam.get("resolution", "1920x1080")
            )

    # Process callback
    def on_frame_result(result):
        # Publish detections
        publisher.process_detections(
            camera_name=result.camera_name,
            detections=result.detections
        )

        # Log significant detections
        for det in result.detections:
            if object_filter.should_include(det.class_name, det.class_id, det.confidence):
                event = DetectionEvent(
                    event_type="detection",
                    timestamp=datetime.now().isoformat(),
                    camera_name=result.camera_name,
                    track_id=det.track_id,
                    class_name=det.class_name,
                    class_id=det.class_id,
                    confidence=det.confidence,
                    bbox={
                        "x1": det.bbox.x1, "y1": det.bbox.y1,
                        "x2": det.bbox.x2, "y2": det.bbox.y2
                    }
                )
                log_detection(event)

    live.on_result(on_frame_result)

    # Start
    print("\n" + "-"*60)
    print("Starting sky watcher...")
    print("Press 'q' to quit, 's' for status")
    print("-"*60 + "\n")

    with live:
        cv2.namedWindow("Sky Watcher", cv2.WINDOW_NORMAL)
        cv2.resizeWindow("Sky Watcher", 1280, 720)

        start_time = time.time()
        frame_count = 0
        detection_count = 0

        try:
            while True:
                result = live.get_result(timeout=0.1)

                if result:
                    frame_count += 1
                    detection_count += len(result.detections)

                    if result.image is not None:
                        # Add info overlay
                        elapsed = time.time() - start_time
                        info = (
                            f"Uptime: {int(elapsed)}s | "
                            f"Frames: {frame_count} | "
                            f"Detections: {detection_count} | "
                            f"FPS: {frame_count/elapsed:.1f}"
                        )
                        cv2.putText(
                            result.image, info, (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2
                        )

                        # Highlight airplanes
                        for det in result.detections:
                            if det.class_id == 4 or "airplane" in det.class_name.lower():
                                x1, y1 = int(det.bbox.x1), int(det.bbox.y1)
                                cv2.putText(
                                    result.image, "AIRCRAFT", (x1, y1 - 30),
                                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 3
                                )

                        cv2.imshow("Sky Watcher", result.image)

                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    break
                elif key == ord('s'):
                    elapsed = time.time() - start_time
                    print(f"\n--- STATUS ---")
                    print(f"Uptime: {elapsed:.1f}s")
                    print(f"Frames processed: {frame_count}")
                    print(f"Total detections: {detection_count}")
                    print(f"Average FPS: {frame_count/elapsed:.1f}")
                    print(f"Pipeline stats: {pipeline.stats}")
                    print(f"--------------\n")

        except KeyboardInterrupt:
            print("\nShutting down...")

    cv2.destroyAllWindows()
    publisher.stop()

    # Final stats
    elapsed = time.time() - start_time
    print("\n" + "="*60)
    print("SESSION COMPLETE")
    print("="*60)
    print(f"Duration: {elapsed:.1f} seconds")
    print(f"Frames processed: {frame_count}")
    print(f"Detections: {detection_count}")
    print(f"Average FPS: {frame_count/elapsed:.1f}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
