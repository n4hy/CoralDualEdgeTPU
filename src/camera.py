"""Network camera interface for AXIS, Empire Tech, and other RTSP cameras.

Supports:
- AXIS M3057-PLVE MK II panoramic cameras
- Empire Tech PTZ425DB-AT PTZ cameras (4MP 25x zoom)
- Generic RTSP/ONVIF streams
- PTZ control (pan, tilt, zoom)
- Frame buffering for consistent inference rates
"""

import threading
import time
from dataclasses import dataclass
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
                raise RuntimeError("Cannot start: camera not connected")

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


class EmpireTechPTZ(AxisCamera):
    """Interface for Empire Tech PTZ cameras (PTZ425DB-AT and similar).

    Empire Tech PTZ425DB-AT specs:
    - 4MP 1/2.8" STARVIS CMOS sensor
    - 25x optical zoom (5mm-125mm)
    - H.264/H.265 encoding
    - RTSP streaming
    - Auto-tracking, perimeter protection
    - IR distance 100m

    RTSP URL format:
    - Main stream: rtsp://<ip>/cam/realmonitor?channel=1&subtype=0
    - Sub stream:  rtsp://<ip>/cam/realmonitor?channel=1&subtype=1
    """

    # Stream types
    MAIN_STREAM = "subtype=0"  # 4MP full resolution
    SUB_STREAM = "subtype=1"   # Lower resolution for preview

    # Resolution presets
    RES_4MP = "2560x1440"
    RES_1080P = "1920x1080"
    RES_720P = "1280x720"

    def __init__(self, config: CameraConfig):
        super().__init__(config)
        self._ptz_session = None

    @classmethod
    def create_rtsp_url(cls, ip: str, username: str = "", password: str = "",
                        channel: int = 1, subtype: int = 0) -> str:
        """Build RTSP URL for Empire Tech camera.

        Args:
            ip: Camera IP address
            username: Camera username
            password: Camera password
            channel: Video channel (usually 1)
            subtype: 0 for main stream (4MP), 1 for sub stream

        Returns:
            Complete RTSP URL
        """
        if username and password:
            return f"rtsp://{username}:{password}@{ip}/cam/realmonitor?channel={channel}&subtype={subtype}"
        return f"rtsp://{ip}/cam/realmonitor?channel={channel}&subtype={subtype}"

    def ptz_move(self, pan: float = 0, tilt: float = 0, zoom: float = 0,
                 speed: float = 0.5) -> bool:
        """Move the PTZ camera.

        Args:
            pan: Pan speed (-1.0 to 1.0, negative=left, positive=right)
            tilt: Tilt speed (-1.0 to 1.0, negative=down, positive=up)
            zoom: Zoom speed (-1.0 to 1.0, negative=wide, positive=tele)
            speed: Movement speed multiplier (0.0 to 1.0)

        Returns:
            True if command sent successfully
        """
        # CGI command for PTZ control (Dahua-compatible)
        # Most Empire Tech cameras use Dahua protocol
        import requests
        from requests.auth import HTTPDigestAuth

        try:
            # Continuous move command
            url = f"http://{self._get_ip()}/cgi-bin/ptz.cgi"
            # Camera requires speed in both arg1 and arg2
            move_speed = max(int(abs(pan) * speed * 8), int(abs(tilt) * speed * 8), 1)
            params = {
                "action": "start",
                "channel": 1,
                "code": self._get_ptz_code(pan, tilt, zoom),
                "arg1": move_speed,
                "arg2": move_speed,
                "arg3": 0
            }

            auth = None
            if self.config.username:
                auth = HTTPDigestAuth(self.config.username, self.config.password)

            response = requests.get(url, params=params, auth=auth, timeout=2)
            return response.status_code == 200

        except Exception as e:
            print(f"[{self.config.name}] PTZ error: {e}")
            return False

    def ptz_stop(self) -> bool:
        """Stop all PTZ movement including zoom."""
        import requests
        from requests.auth import HTTPDigestAuth

        try:
            url = f"http://{self._get_ip()}/cgi-bin/ptz.cgi"

            auth = None
            if self.config.username:
                auth = HTTPDigestAuth(self.config.username, self.config.password)

            # Stop pan/tilt
            params = {
                "action": "start",
                "channel": 1,
                "code": "Up",
                "arg1": 0,
                "arg2": 0,
                "arg3": 0
            }
            r1 = requests.get(url, params=params, auth=auth, timeout=2)

            # Stop zoom
            params["code"] = "ZoomTele"
            r2 = requests.get(url, params=params, auth=auth, timeout=2)

            return r1.status_code == 200 and r2.status_code == 200

        except Exception as e:
            print(f"[{self.config.name}] PTZ stop error: {e}")
            return False

    def ptz_goto_preset(self, preset: int) -> bool:
        """Move to a saved preset position.

        Args:
            preset: Preset number (1-255)

        Returns:
            True if command sent successfully
        """
        import requests
        from requests.auth import HTTPDigestAuth

        try:
            url = f"http://{self._get_ip()}/cgi-bin/ptz.cgi"
            params = {
                "action": "start",
                "channel": 1,
                "code": "GotoPreset",
                "arg1": 0,
                "arg2": preset,
                "arg3": 0
            }

            auth = None
            if self.config.username:
                auth = HTTPDigestAuth(self.config.username, self.config.password)

            response = requests.get(url, params=params, auth=auth, timeout=5)
            return response.status_code == 200

        except Exception as e:
            print(f"[{self.config.name}] PTZ preset error: {e}")
            return False

    def ptz_set_preset(self, preset: int) -> bool:
        """Save current position as preset.

        Args:
            preset: Preset number (1-255)

        Returns:
            True if command sent successfully
        """
        import requests
        from requests.auth import HTTPDigestAuth

        try:
            url = f"http://{self._get_ip()}/cgi-bin/ptz.cgi"
            params = {
                "action": "start",
                "channel": 1,
                "code": "SetPreset",
                "arg1": 0,
                "arg2": preset,
                "arg3": 0
            }

            auth = None
            if self.config.username:
                auth = HTTPDigestAuth(self.config.username, self.config.password)

            response = requests.get(url, params=params, auth=auth, timeout=2)
            return response.status_code == 200

        except Exception as e:
            print(f"[{self.config.name}] PTZ set preset error: {e}")
            return False

    def zoom_to(self, zoom_level: float) -> bool:
        """Set absolute zoom level.

        Args:
            zoom_level: Zoom level (0.0 = wide, 1.0 = full tele 25x)

        Returns:
            True if command sent successfully
        """
        # Use relative zoom to approximate absolute
        # Full zoom range on PTZ425DB-AT is 25x
        import requests
        from requests.auth import HTTPDigestAuth

        try:
            url = f"http://{self._get_ip()}/cgi-bin/ptz.cgi"
            params = {
                "action": "start",
                "channel": 1,
                "code": "ZoomTele" if zoom_level > 0.5 else "ZoomWide",
                "arg1": int(zoom_level * 8),
                "arg2": 0,
                "arg3": 0
            }

            auth = None
            if self.config.username:
                auth = HTTPDigestAuth(self.config.username, self.config.password)

            response = requests.get(url, params=params, auth=auth, timeout=2)
            return response.status_code == 200

        except Exception as e:
            print(f"[{self.config.name}] Zoom error: {e}")
            return False

    def get_position(self) -> Optional[tuple]:
        """Get current PTZ position.

        Returns:
            Tuple of (azimuth, elevation, zoom) in degrees, or None on error.
            Azimuth: 0-360 degrees
            Elevation: -5 to 90 degrees (0 = horizontal, 90 = straight up)
            Zoom: 1-25x optical zoom
        """
        import requests
        from requests.auth import HTTPDigestAuth

        try:
            url = f"http://{self._get_ip()}/cgi-bin/ptz.cgi"
            params = {"action": "getStatus", "channel": 1}

            auth = None
            if self.config.username:
                auth = HTTPDigestAuth(self.config.username, self.config.password)

            response = requests.get(url, params=params, auth=auth, timeout=2)
            if response.status_code != 200:
                return None

            # Parse response: status.Postion[0]=123.45 etc
            az, el, zoom = 0.0, 0.0, 1.0
            for line in response.text.split('\n'):
                if 'Postion[0]=' in line:
                    az = float(line.split('=')[1])
                elif 'Postion[1]=' in line:
                    el = float(line.split('=')[1])
                elif 'ZoomValue=' in line:
                    zoom = float(line.split('=')[1]) / 100.0  # Convert to 1-25x

            return (az, el, zoom)

        except Exception as e:
            print(f"[{self.config.name}] Get position error: {e}")
            return None

    def is_moving(self) -> bool:
        """Check if camera is currently moving.

        Returns:
            True if camera is in motion, False if idle
        """
        import requests
        from requests.auth import HTTPDigestAuth

        try:
            url = f"http://{self._get_ip()}/cgi-bin/ptz.cgi"
            params = {"action": "getStatus", "channel": 1}

            auth = None
            if self.config.username:
                auth = HTTPDigestAuth(self.config.username, self.config.password)

            response = requests.get(url, params=params, auth=auth, timeout=2)
            if response.status_code != 200:
                return False

            return "MoveStatus=Moving" in response.text

        except Exception:
            return False

    def wait_for_idle(self, timeout: float = 30.0) -> bool:
        """Wait for camera to stop moving based on MoveStatus.

        Args:
            timeout: Maximum seconds to wait

        Returns:
            True if camera is idle, False if timeout
        """
        import time
        start = time.time()
        was_moving = False

        while time.time() - start < timeout:
            moving = self.is_moving()
            if moving:
                was_moving = True
            elif was_moving:
                # Was moving, now idle - movement complete
                return True
            elif time.time() - start > 1.0:
                # Never started moving after 1s - command may have failed
                return False
            time.sleep(0.1)
        return False

    def goto_position(self, azimuth: float, elevation: float,
                      zoom: Optional[float] = None,
                      wait: bool = True) -> bool:
        """Move to absolute azimuth/elevation/zoom using PositionABS command.

        Args:
            azimuth: Target azimuth in degrees (0-360)
            elevation: Target elevation in degrees (0-90, 0=horizon, 90=up)
            zoom: Target zoom level (1.0-25.0x), or None to keep current zoom
            wait: If True, wait for camera to reach position

        Returns:
            True if command sent (and position reached if wait=True)
        """
        import requests
        from requests.auth import HTTPDigestAuth

        try:
            url = f"http://{self._get_ip()}/cgi-bin/ptz.cgi"
            params = {
                "action": "start",
                "channel": 1,
                "code": "PositionABS",
                "arg1": int(azimuth),
                "arg2": int(elevation),
                "arg3": int(zoom * 100) if zoom is not None else 0
            }

            auth = None
            if self.config.username:
                auth = HTTPDigestAuth(self.config.username, self.config.password)

            response = requests.get(url, params=params, auth=auth, timeout=5)
            if response.status_code != 200:
                return False

            if wait:
                return self.wait_for_idle()
            return True

        except Exception as e:
            print(f"[{self.config.name}] PositionABS error: {e}")
            return False

    def _get_ip(self) -> str:
        """Extract IP from RTSP URL."""
        url = self.config.rtsp_url
        # Handle rtsp://user:pass@ip/... or rtsp://ip/...
        if "@" in url:
            return url.split("@")[1].split("/")[0].split(":")[0]
        else:
            return url.replace("rtsp://", "").split("/")[0].split(":")[0]

    def _get_ptz_code(self, pan: float, tilt: float, zoom: float) -> str:
        """Convert movement values to PTZ command code."""
        if zoom > 0.1:
            return "ZoomTele"
        elif zoom < -0.1:
            return "ZoomWide"
        elif pan > 0.1 and tilt > 0.1:
            return "RightUp"
        elif pan > 0.1 and tilt < -0.1:
            return "RightDown"
        elif pan < -0.1 and tilt > 0.1:
            return "LeftUp"
        elif pan < -0.1 and tilt < -0.1:
            return "LeftDown"
        elif pan > 0.1:
            return "Right"
        elif pan < -0.1:
            return "Left"
        elif tilt > 0.1:
            return "Up"
        elif tilt < -0.1:
            return "Down"
        return "Up"  # Default


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

    def add_empiretech_ptz(self, name: str, ip: str,
                          username: str = "", password: str = "",
                          main_stream: bool = True) -> EmpireTechPTZ:
        """Add an Empire Tech PTZ camera (PTZ425DB-AT or similar).

        Args:
            name: Friendly name for the camera
            ip: Camera IP address
            username: Camera username
            password: Camera password
            main_stream: True for 4MP main stream, False for sub stream

        Returns:
            EmpireTechPTZ camera instance with PTZ control
        """
        subtype = 0 if main_stream else 1
        url = EmpireTechPTZ.create_rtsp_url(ip, username, password, subtype=subtype)
        config = CameraConfig(
            name=name,
            rtsp_url=url,
            username=username,
            password=password,
            resolution=(2560, 1440) if main_stream else (1280, 720)
        )
        camera = EmpireTechPTZ(config)
        self.cameras[name] = camera
        return camera

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
