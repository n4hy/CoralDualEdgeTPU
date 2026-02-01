"""Network camera interface for AXIS and other RTSP cameras.

Supports:
- AXIS M3057-PLVE MK II panoramic cameras
- Generic RTSP streams
- Frame buffering for consistent inference rates
"""

import threading
import time
from dataclasses import dataclass, field
from queue import Queue, Empty
from typing import Optional, Callable

import cv2
import numpy as np


@dataclass
class CameraConfig:
    """Configuration for a network camera."""
    name: str
    rtsp_url: str
    username: str = ""
    password: str = ""
    resolution: tuple[int, int] = (1920, 1080)
    fps: int = 15
    buffer_size: int = 2
    reconnect_delay: float = 5.0

    def get_auth_url(self) -> str:
        """Get RTSP URL with embedded credentials."""
        if self.username and self.password:
            # rtsp://user:pass@host/path
            if "://" in self.rtsp_url:
                proto, rest = self.rtsp_url.split("://", 1)
                return f"{proto}://{self.username}:{self.password}@{rest}"
        return self.rtsp_url


@dataclass
class Frame:
    """A captured frame with metadata."""
    image: np.ndarray
    timestamp: float
    camera_name: str
    frame_number: int


class AxisCamera:
    """Interface for AXIS network cameras.

    AXIS M3057-PLVE MK II specifics:
    - 6MP panoramic sensor
    - Multiple view modes: panoramic, double-panorama, quad, view-area
    - RTSP streaming with H.264/H.265

    Typical RTSP URLs for AXIS cameras:
    - Main stream: rtsp://<ip>/axis-media/media.amp
    - Lower res: rtsp://<ip>/axis-media/media.amp?resolution=640x480
    - Specific view: rtsp://<ip>/axis-media/media.amp?camera=1
    """

    # AXIS M3057 view area mappings
    VIEW_PANORAMIC = "camera=1"
    VIEW_DOUBLE_PANORAMA = "camera=2"
    VIEW_QUAD = "camera=3"
    VIEW_AREA_1 = "camera=4"
    VIEW_AREA_2 = "camera=5"
    VIEW_AREA_3 = "camera=6"
    VIEW_AREA_4 = "camera=7"

    def __init__(self, config: CameraConfig):
        self.config = config
        self._cap: Optional[cv2.VideoCapture] = None
        self._frame_queue: Queue[Frame] = Queue(maxsize=config.buffer_size)
        self._running = False
        self._thread: Optional[threading.Thread] = None
        self._frame_count = 0
        self._lock = threading.Lock()
        self._last_frame: Optional[Frame] = None
        self._on_frame_callback: Optional[Callable[[Frame], None]] = None

    @classmethod
    def create_axis_url(cls, ip: str, view: str = "", resolution: str = "",
                        username: str = "", password: str = "") -> str:
        """Build RTSP URL for AXIS camera.

        Args:
            ip: Camera IP address
            view: View mode (use VIEW_* constants)
            resolution: e.g., "1920x1080", "1280x720", "640x480"
            username: Camera username
            password: Camera password

        Returns:
            Complete RTSP URL
        """
        params = []
        if view:
            params.append(view)
        if resolution:
            params.append(f"resolution={resolution}")

        url = f"rtsp://{ip}/axis-media/media.amp"
        if params:
            url += "?" + "&".join(params)

        if username and password:
            url = f"rtsp://{username}:{password}@{ip}/axis-media/media.amp"
            if params:
                url += "?" + "&".join(params)

        return url

    def connect(self) -> bool:
        """Connect to the camera stream."""
        url = self.config.get_auth_url()

        # OpenCV RTSP settings for stability
        self._cap = cv2.VideoCapture(url, cv2.CAP_FFMPEG)
        self._cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

        if not self._cap.isOpened():
            print(f"[{self.config.name}] Failed to connect to {self.config.rtsp_url}")
            return False

        # Get actual properties
        actual_w = int(self._cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        actual_h = int(self._cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        actual_fps = self._cap.get(cv2.CAP_PROP_FPS)

        print(f"[{self.config.name}] Connected: {actual_w}x{actual_h} @ {actual_fps:.1f}fps")
        return True

    def disconnect(self):
        """Disconnect from the camera."""
        self._running = False
        if self._thread:
            self._thread.join(timeout=2.0)
        if self._cap:
            self._cap.release()
            self._cap = None
        print(f"[{self.config.name}] Disconnected")

    def start(self, on_frame: Optional[Callable[[Frame], None]] = None):
        """Start background frame capture.

        Args:
            on_frame: Optional callback for each new frame
        """
        if self._running:
            return

        if not self._cap or not self._cap.isOpened():
            if not self.connect():
                raise RuntimeError(f"Cannot start: camera not connected")

        self._on_frame_callback = on_frame
        self._running = True
        self._thread = threading.Thread(target=self._capture_loop, daemon=True)
        self._thread.start()
        print(f"[{self.config.name}] Capture started")

    def stop(self):
        """Stop background capture."""
        self._running = False
        if self._thread:
            self._thread.join(timeout=2.0)
            self._thread = None
        print(f"[{self.config.name}] Capture stopped")

    def _capture_loop(self):
        """Background frame capture loop."""
        consecutive_failures = 0
        max_failures = 10

        while self._running:
            if not self._cap or not self._cap.isOpened():
                print(f"[{self.config.name}] Reconnecting...")
                time.sleep(self.config.reconnect_delay)
                if not self.connect():
                    consecutive_failures += 1
                    if consecutive_failures > max_failures:
                        print(f"[{self.config.name}] Too many failures, stopping")
                        break
                    continue
                consecutive_failures = 0

            ret, image = self._cap.read()

            if not ret:
                consecutive_failures += 1
                if consecutive_failures > max_failures:
                    # Try to reconnect
                    self._cap.release()
                    self._cap = None
                continue

            consecutive_failures = 0
            self._frame_count += 1

            frame = Frame(
                image=image,
                timestamp=time.time(),
                camera_name=self.config.name,
                frame_number=self._frame_count
            )

            with self._lock:
                self._last_frame = frame

            # Update queue (drop old frames if full)
            if self._frame_queue.full():
                try:
                    self._frame_queue.get_nowait()
                except Empty:
                    pass
            self._frame_queue.put(frame)

            # Callback
            if self._on_frame_callback:
                try:
                    self._on_frame_callback(frame)
                except Exception as e:
                    print(f"[{self.config.name}] Callback error: {e}")

    def get_frame(self, timeout: float = 1.0) -> Optional[Frame]:
        """Get next frame from buffer.

        Args:
            timeout: Max time to wait for a frame

        Returns:
            Frame or None if timeout
        """
        try:
            return self._frame_queue.get(timeout=timeout)
        except Empty:
            return None

    def get_latest_frame(self) -> Optional[Frame]:
        """Get most recent frame (non-blocking)."""
        with self._lock:
            return self._last_frame

    @property
    def is_running(self) -> bool:
        return self._running

    @property
    def frame_count(self) -> int:
        return self._frame_count


class MultiCameraManager:
    """Manages multiple network cameras."""

    def __init__(self):
        self.cameras: dict[str, AxisCamera] = {}
        self._running = False

    def add_camera(self, config: CameraConfig) -> AxisCamera:
        """Add a camera to the manager."""
        camera = AxisCamera(config)
        self.cameras[config.name] = camera
        return camera

    def add_axis_camera(self, name: str, ip: str,
                        username: str = "", password: str = "",
                        view: str = "", resolution: str = "1280x720") -> AxisCamera:
        """Convenience method to add an AXIS camera.

        Args:
            name: Friendly name for the camera
            ip: Camera IP address
            username: Camera username
            password: Camera password
            view: View mode for panoramic cameras
            resolution: Stream resolution
        """
        url = AxisCamera.create_axis_url(ip, view, resolution, username, password)
        config = CameraConfig(
            name=name,
            rtsp_url=url,
            username=username,
            password=password
        )
        return self.add_camera(config)

    def start_all(self):
        """Start all cameras."""
        self._running = True
        for name, camera in self.cameras.items():
            try:
                camera.start()
            except Exception as e:
                print(f"Failed to start {name}: {e}")

    def stop_all(self):
        """Stop all cameras."""
        self._running = False
        for camera in self.cameras.values():
            camera.stop()

    def get_frames(self, timeout: float = 0.1) -> dict[str, Frame]:
        """Get latest frame from each camera.

        Returns:
            Dict mapping camera name to latest frame
        """
        frames = {}
        for name, camera in self.cameras.items():
            frame = camera.get_latest_frame()
            if frame:
                frames[name] = frame
        return frames

    def __enter__(self):
        self.start_all()
        return self

    def __exit__(self, *args):
        self.stop_all()
