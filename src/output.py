"""MQTT and Webhook output for detection events.

Publishes detection/tracking events for:
- Airplanes
- Satellites (solar-illuminated at night)
- Custom object classes

Supports:
- MQTT with configurable topics
- HTTP webhooks (POST JSON)
- Event filtering and rate limiting
"""

import json
import threading
import time
from dataclasses import dataclass, asdict
from datetime import datetime
from queue import Queue, Empty
from typing import Optional
import uuid

import requests


@dataclass
class DetectionEvent:
    """A detection event for external publishing."""
    event_type: str  # "detection", "track_start", "track_end"
    timestamp: str
    camera_name: str
    track_id: int
    class_name: str
    class_id: int
    confidence: float
    bbox: dict  # {x1, y1, x2, y2}
    classification: Optional[str] = None
    classification_confidence: float = 0.0
    velocity: Optional[tuple] = None
    track_age: int = 0

    def to_dict(self) -> dict:
        return asdict(self)

    def to_json(self) -> str:
        return json.dumps(self.to_dict())


class ObjectFilter:
    """Filter detections by object class."""

    # Default classes of interest for sky watching
    AIRPLANE_CLASSES = {
        "airplane", "aeroplane", "aircraft", "plane", "jet",
        # COCO class
        4,  # airplane in COCO
    }

    SATELLITE_CLASSES = {
        "satellite", "spacecraft", "iss",
        # Note: Standard models don't have satellite class
        # Will need custom model or bright-point detection
    }

    # Bright point detection for satellites
    # Satellites appear as bright moving points at night

    def __init__(self,
                 include_classes: Optional[set] = None,
                 exclude_classes: Optional[set] = None,
                 min_confidence: float = 0.5,
                 airplane_mode: bool = True,
                 satellite_mode: bool = True):
        """
        Args:
            include_classes: Set of class names/IDs to include (None = all)
            exclude_classes: Set of class names/IDs to exclude
            min_confidence: Minimum confidence threshold
            airplane_mode: Include airplane-related classes
            satellite_mode: Enable satellite detection logic
        """
        self.include_classes = include_classes or set()
        self.exclude_classes = exclude_classes or set()
        self.min_confidence = min_confidence
        self.airplane_mode = airplane_mode
        self.satellite_mode = satellite_mode

        # Auto-add airplane classes if enabled
        if airplane_mode:
            self.include_classes.update(self.AIRPLANE_CLASSES)

    def should_include(self, class_name: str, class_id: int,
                       confidence: float) -> bool:
        """Check if detection should be included."""
        if confidence < self.min_confidence:
            return False

        # Check excludes first
        if class_name in self.exclude_classes or class_id in self.exclude_classes:
            return False

        # If includes specified, must match
        if self.include_classes:
            return (class_name in self.include_classes or
                    class_id in self.include_classes or
                    class_name.lower() in self.include_classes)

        return True


class MQTTOutput:
    """MQTT publisher for detection events."""

    def __init__(self,
                 broker: str,
                 port: int = 1883,
                 username: Optional[str] = None,
                 password: Optional[str] = None,
                 topic_prefix: str = "coral/detections",
                 client_id: Optional[str] = None,
                 qos: int = 1):
        """
        Args:
            broker: MQTT broker hostname/IP
            port: MQTT broker port
            username: Optional username
            password: Optional password
            topic_prefix: Base topic (events published to prefix/camera/class)
            client_id: MQTT client ID
            qos: Quality of Service level (0, 1, or 2)
        """
        self.broker = broker
        self.port = port
        self.username = username
        self.password = password
        self.topic_prefix = topic_prefix
        self.client_id = client_id or f"coral-tpu-{uuid.uuid4().hex[:8]}"
        self.qos = qos

        self._client = None
        self._connected = False
        self._queue: Queue = Queue(maxsize=1000)
        self._thread: Optional[threading.Thread] = None
        self._running = False

    def connect(self) -> bool:
        """Connect to MQTT broker."""
        try:
            import paho.mqtt.client as mqtt
        except ImportError:
            print("ERROR: paho-mqtt not installed. Run: pip install paho-mqtt")
            return False

        def on_connect(client, userdata, flags, rc):
            if rc == 0:
                self._connected = True
                print(f"MQTT connected to {self.broker}:{self.port}")
            else:
                print(f"MQTT connection failed: rc={rc}")

        def on_disconnect(client, userdata, rc):
            self._connected = False
            print(f"MQTT disconnected: rc={rc}")

        self._client = mqtt.Client(client_id=self.client_id)
        self._client.on_connect = on_connect
        self._client.on_disconnect = on_disconnect

        if self.username:
            self._client.username_pw_set(self.username, self.password)

        try:
            self._client.connect(self.broker, self.port, keepalive=60)
            self._client.loop_start()

            # Wait for connection
            for _ in range(10):
                if self._connected:
                    break
                time.sleep(0.1)

            return self._connected

        except Exception as e:
            print(f"MQTT connection error: {e}")
            return False

    def disconnect(self):
        """Disconnect from broker."""
        self._running = False
        if self._thread:
            self._thread.join(timeout=2.0)
        if self._client:
            self._client.loop_stop()
            self._client.disconnect()
        self._connected = False

    def start_async(self):
        """Start async publishing thread."""
        self._running = True
        self._thread = threading.Thread(target=self._publish_loop, daemon=True)
        self._thread.start()

    def _publish_loop(self):
        """Background publishing loop."""
        while self._running:
            try:
                event = self._queue.get(timeout=0.5)
                self._publish_event(event)
            except Empty:
                continue

    def _publish_event(self, event: DetectionEvent):
        """Publish event to MQTT."""
        if not self._connected or not self._client:
            return

        # Build topic: prefix/camera/class_name/event_type
        topic = f"{self.topic_prefix}/{event.camera_name}/{event.class_name}/{event.event_type}"

        try:
            self._client.publish(
                topic,
                event.to_json(),
                qos=self.qos
            )
        except Exception as e:
            print(f"MQTT publish error: {e}")

    def publish(self, event: DetectionEvent):
        """Queue event for publishing."""
        if not self._queue.full():
            self._queue.put(event)

    def publish_sync(self, event: DetectionEvent):
        """Publish event synchronously."""
        self._publish_event(event)


class WebhookOutput:
    """HTTP webhook publisher for detection events."""

    def __init__(self,
                 url: str,
                 headers: Optional[dict] = None,
                 timeout: float = 5.0,
                 retry_count: int = 2,
                 batch_size: int = 1,
                 batch_timeout: float = 0.5):
        """
        Args:
            url: Webhook endpoint URL
            headers: Additional HTTP headers
            timeout: Request timeout in seconds
            retry_count: Number of retries on failure
            batch_size: Events to batch before sending
            batch_timeout: Max time to wait for batch
        """
        self.url = url
        self.headers = headers or {"Content-Type": "application/json"}
        self.timeout = timeout
        self.retry_count = retry_count
        self.batch_size = batch_size
        self.batch_timeout = batch_timeout

        self._queue: Queue = Queue(maxsize=1000)
        self._thread: Optional[threading.Thread] = None
        self._running = False
        self._session = requests.Session()
        self._session.headers.update(self.headers)

    def start(self):
        """Start async publishing."""
        self._running = True
        self._thread = threading.Thread(target=self._publish_loop, daemon=True)
        self._thread.start()

    def stop(self):
        """Stop publishing."""
        self._running = False
        if self._thread:
            self._thread.join(timeout=2.0)

    def _publish_loop(self):
        """Background publishing loop with batching."""
        batch = []
        batch_start = time.time()

        while self._running:
            try:
                event = self._queue.get(timeout=0.1)
                batch.append(event)
            except Empty:
                pass

            # Send batch if full or timeout
            should_send = (
                len(batch) >= self.batch_size or
                (batch and time.time() - batch_start > self.batch_timeout)
            )

            if should_send and batch:
                self._send_batch(batch)
                batch = []
                batch_start = time.time()

        # Send remaining
        if batch:
            self._send_batch(batch)

    def _send_batch(self, events: list[DetectionEvent]):
        """Send a batch of events."""
        payload = {
            "events": [e.to_dict() for e in events],
            "count": len(events),
            "timestamp": datetime.now().isoformat()
        }

        for attempt in range(self.retry_count + 1):
            try:
                response = self._session.post(
                    self.url,
                    json=payload,
                    timeout=self.timeout
                )

                if response.status_code < 300:
                    return True
                else:
                    print(f"Webhook error {response.status_code}: {response.text[:100]}")

            except requests.exceptions.Timeout:
                print(f"Webhook timeout (attempt {attempt + 1})")
            except requests.exceptions.RequestException as e:
                print(f"Webhook error: {e}")

            if attempt < self.retry_count:
                time.sleep(0.5 * (attempt + 1))

        return False

    def publish(self, event: DetectionEvent):
        """Queue event for publishing."""
        if not self._queue.full():
            self._queue.put(event)

    def publish_sync(self, event: DetectionEvent):
        """Publish event synchronously."""
        self._send_batch([event])


class EventPublisher:
    """Combined event publisher with filtering and rate limiting."""

    def __init__(self,
                 object_filter: Optional[ObjectFilter] = None,
                 rate_limit_per_track: float = 1.0):
        """
        Args:
            object_filter: Filter for which detections to publish
            rate_limit_per_track: Minimum seconds between events for same track
        """
        self.filter = object_filter or ObjectFilter()
        self.rate_limit = rate_limit_per_track

        self._outputs: list = []
        self._last_event_time: dict[tuple, float] = {}  # (camera, track_id) -> timestamp
        self._active_tracks: dict[tuple, DetectionEvent] = {}

    def add_mqtt(self, broker: str, **kwargs) -> MQTTOutput:
        """Add MQTT output."""
        mqtt = MQTTOutput(broker, **kwargs)
        if mqtt.connect():
            mqtt.start_async()
            self._outputs.append(mqtt)
        return mqtt

    def add_webhook(self, url: str, **kwargs) -> WebhookOutput:
        """Add webhook output."""
        webhook = WebhookOutput(url, **kwargs)
        webhook.start()
        self._outputs.append(webhook)
        return webhook

    def process_detections(self, camera_name: str, detections: list,
                           tracks: list = None):
        """Process detection results and publish events.

        Args:
            camera_name: Source camera name
            detections: List of Detection objects from pipeline
            tracks: Optional list of Track objects
        """
        current_time = time.time()
        current_track_ids = set()

        for det in detections:
            # Apply filter
            if not self.filter.should_include(
                det.class_name, det.class_id, det.confidence
            ):
                continue

            track_key = (camera_name, det.track_id)
            current_track_ids.add(track_key)

            # Rate limiting
            last_time = self._last_event_time.get(track_key, 0)
            if current_time - last_time < self.rate_limit:
                continue

            # Determine event type
            if track_key not in self._active_tracks:
                event_type = "track_start"
            else:
                event_type = "detection"

            event = DetectionEvent(
                event_type=event_type,
                timestamp=datetime.now().isoformat(),
                camera_name=camera_name,
                track_id=det.track_id,
                class_name=det.class_name,
                class_id=det.class_id,
                confidence=det.confidence,
                bbox={
                    "x1": det.bbox.x1,
                    "y1": det.bbox.y1,
                    "x2": det.bbox.x2,
                    "y2": det.bbox.y2
                },
                classification=det.classification,
                classification_confidence=det.classification_confidence
            )

            self._publish(event)
            self._last_event_time[track_key] = current_time
            self._active_tracks[track_key] = event

        # Detect track ends
        ended_tracks = set(self._active_tracks.keys()) - current_track_ids
        for track_key in ended_tracks:
            if track_key[0] == camera_name:  # Same camera
                old_event = self._active_tracks.pop(track_key)

                end_event = DetectionEvent(
                    event_type="track_end",
                    timestamp=datetime.now().isoformat(),
                    camera_name=camera_name,
                    track_id=old_event.track_id,
                    class_name=old_event.class_name,
                    class_id=old_event.class_id,
                    confidence=old_event.confidence,
                    bbox=old_event.bbox
                )

                self._publish(end_event)
                self._last_event_time.pop(track_key, None)

    def _publish(self, event: DetectionEvent):
        """Publish to all outputs."""
        for output in self._outputs:
            output.publish(event)

    def stop(self):
        """Stop all outputs."""
        for output in self._outputs:
            if hasattr(output, 'disconnect'):
                output.disconnect()
            elif hasattr(output, 'stop'):
                output.stop()


# Satellite detection utilities
class SatelliteDetector:
    """Helper for detecting solar-illuminated satellites at night.

    Satellites appear as bright, steadily moving points of light.
    Detection strategy:
    1. Threshold for bright pixels
    2. Track motion across frames
    3. Filter by:
       - Consistent velocity (not blinking = plane)
       - Not in known star positions
       - Angular velocity consistent with LEO/MEO orbits
    """

    def __init__(self,
                 brightness_threshold: int = 240,
                 min_velocity: float = 0.5,  # degrees/sec
                 max_velocity: float = 2.0,  # degrees/sec for LEO
                 min_track_length: int = 10):  # frames
        self.brightness_threshold = brightness_threshold
        self.min_velocity = min_velocity
        self.max_velocity = max_velocity
        self.min_track_length = min_track_length

        self._point_tracks: dict = {}

    def detect_bright_points(self, frame_gray) -> list[tuple[int, int]]:
        """Find bright point sources in frame.

        Args:
            frame_gray: Grayscale frame

        Returns:
            List of (x, y) coordinates of bright points
        """
        import cv2

        # Threshold for bright pixels
        _, thresh = cv2.threshold(
            frame_gray, self.brightness_threshold, 255, cv2.THRESH_BINARY
        )

        # Find contours (bright spots)
        contours, _ = cv2.findContours(
            thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )

        points = []
        for contour in contours:
            # Get centroid
            M = cv2.moments(contour)
            if M["m00"] > 0:
                cx = int(M["m10"] / M["m00"])
                cy = int(M["m01"] / M["m00"])

                # Filter by size (satellites are point sources)
                area = cv2.contourArea(contour)
                if area < 100:  # Small bright spot
                    points.append((cx, cy))

        return points

    def is_satellite_candidate(self, velocity: tuple[float, float],
                               pixels_per_degree: float = 10.0) -> bool:
        """Check if velocity matches satellite motion.

        Args:
            velocity: (vx, vy) in pixels per frame
            pixels_per_degree: Camera calibration

        Returns:
            True if velocity consistent with satellite
        """
        # Convert to angular velocity (rough estimate)
        speed_pixels = (velocity[0]**2 + velocity[1]**2)**0.5
        speed_deg_per_sec = speed_pixels / pixels_per_degree  # Assuming ~30fps

        return self.min_velocity <= speed_deg_per_sec <= self.max_velocity
