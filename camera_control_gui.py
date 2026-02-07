#!/usr/bin/env python3
"""
PTZ Camera Control GUI with RTSP Streaming

Full-featured camera control application for Empire Tech PTZ cameras.
Designed to run via X11 forwarding from a remote Linux desktop.

Features:
- Live RTSP video streaming with FPS display
- Full PTZ control (pan, tilt, zoom)
- Speed adjustment
- Preset management (1-255)
- Keyboard shortcuts
- Connection status monitoring

Usage:
    ssh -X user@orangepi
    source coral39/bin/activate
    python camera_control_gui.py
"""

import threading
import time
import tkinter as tk
from tkinter import ttk, messagebox
from typing import Optional

import cv2
import numpy as np
from PIL import Image, ImageTk

from src.camera import CameraConfig, EmpireTechPTZ


class VideoStream:
    """Threaded RTSP video capture with frame buffering."""

    def __init__(self):
        self._cap: Optional[cv2.VideoCapture] = None
        self._frame: Optional[np.ndarray] = None
        self._lock = threading.Lock()
        self._running = False
        self._thread: Optional[threading.Thread] = None
        self._connected = False
        self._error: Optional[str] = None
        self._fps = 0.0
        self._frame_count = 0
        self._resolution = (0, 0)

    def start(self, url: str) -> None:
        """Start video capture from RTSP URL."""
        self._running = True
        self._error = None
        self._connected = False
        self._thread = threading.Thread(target=self._capture_loop, args=(url,), daemon=True)
        self._thread.start()

    def stop(self) -> None:
        """Stop video capture."""
        self._running = False
        if self._thread:
            self._thread.join(timeout=3.0)
        with self._lock:
            if self._cap:
                self._cap.release()
            self._cap = None
            self._connected = False
            self._frame = None
            self._fps = 0.0

    def _capture_loop(self, url: str) -> None:
        """Background capture loop."""
        print(f"[VideoStream] Connecting to: {url}")

        try:
            self._cap = cv2.VideoCapture(url, cv2.CAP_FFMPEG)
            self._cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

            if not self._cap.isOpened():
                self._error = "Failed to open RTSP stream"
                print(f"[VideoStream] {self._error}")
                return

            # Get stream properties
            w = int(self._cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            h = int(self._cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            self._resolution = (w, h)
            print(f"[VideoStream] Connected: {w}x{h}")

            self._connected = True
            self._frame_count = 0
            fps_time = time.time()
            fail_count = 0

            while self._running:
                ret, frame = self._cap.read()

                if not ret:
                    fail_count += 1
                    if fail_count > 30:
                        self._error = "Lost connection to camera"
                        print(f"[VideoStream] {self._error}")
                        break
                    time.sleep(0.01)
                    continue

                fail_count = 0
                self._frame_count += 1

                with self._lock:
                    self._frame = frame

                # Update FPS every second
                now = time.time()
                elapsed = now - fps_time
                if elapsed >= 1.0:
                    self._fps = self._frame_count / elapsed
                    self._frame_count = 0
                    fps_time = now

        except Exception as e:
            self._error = str(e)
            print(f"[VideoStream] Error: {e}")
        finally:
            self._connected = False
            print("[VideoStream] Stopped")

    def get_frame(self) -> Optional[np.ndarray]:
        """Get the latest frame (thread-safe)."""
        with self._lock:
            if self._frame is not None:
                return self._frame.copy()
            return None

    @property
    def is_connected(self) -> bool:
        return self._connected

    @property
    def fps(self) -> float:
        return self._fps

    @property
    def resolution(self) -> tuple:
        return self._resolution

    @property
    def error(self) -> Optional[str]:
        return self._error


class PTZCameraGUI:
    """Main GUI application for PTZ camera control."""

    def __init__(self, root: tk.Tk):
        self.root = root
        self.root.title("PTZ Camera Control")
        self.root.geometry("1280x800")
        self.root.minsize(1000, 700)

        # State
        self._camera: Optional[EmpireTechPTZ] = None
        self._video = VideoStream()
        self._update_job: Optional[str] = None

        # Style
        self.style = ttk.Style()
        self.style.theme_use('clam')

        # Build UI
        self._create_widgets()
        self._create_bindings()
        self._start_video_loop()

        print("[GUI] Application started")

    def _create_widgets(self):
        """Create all GUI widgets."""
        # Main container with padding
        main = ttk.Frame(self.root, padding=10)
        main.pack(fill=tk.BOTH, expand=True)

        # === LEFT SIDE: Video Display ===
        video_frame = ttk.LabelFrame(main, text="Live Video", padding=5)
        video_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(0, 10))

        # Video canvas
        self.canvas = tk.Canvas(video_frame, bg='#1a1a1a', highlightthickness=0)
        self.canvas.pack(fill=tk.BOTH, expand=True)

        # Video info bar
        info_bar = ttk.Frame(video_frame)
        info_bar.pack(fill=tk.X, pady=(5, 0))

        self.lbl_fps = ttk.Label(info_bar, text="FPS: --", width=12)
        self.lbl_fps.pack(side=tk.LEFT)

        self.lbl_res = ttk.Label(info_bar, text="Resolution: --", width=20)
        self.lbl_res.pack(side=tk.LEFT, padx=10)

        self.lbl_status = ttk.Label(info_bar, text="● Disconnected", foreground='red')
        self.lbl_status.pack(side=tk.RIGHT)

        # === RIGHT SIDE: Controls ===
        ctrl_frame = ttk.Frame(main, width=320)
        ctrl_frame.pack(side=tk.RIGHT, fill=tk.Y)
        ctrl_frame.pack_propagate(False)

        # --- Connection Section ---
        conn_frame = ttk.LabelFrame(ctrl_frame, text="Connection", padding=10)
        conn_frame.pack(fill=tk.X, pady=(0, 10))

        # IP Address
        row = ttk.Frame(conn_frame)
        row.pack(fill=tk.X, pady=2)
        ttk.Label(row, text="Camera IP:", width=12).pack(side=tk.LEFT)
        self.var_ip = tk.StringVar(value="192.168.1.108")
        ttk.Entry(row, textvariable=self.var_ip, width=18).pack(side=tk.LEFT, padx=5)

        # Username
        row = ttk.Frame(conn_frame)
        row.pack(fill=tk.X, pady=2)
        ttk.Label(row, text="Username:", width=12).pack(side=tk.LEFT)
        self.var_user = tk.StringVar(value="admin")
        ttk.Entry(row, textvariable=self.var_user, width=18).pack(side=tk.LEFT, padx=5)

        # Password
        row = ttk.Frame(conn_frame)
        row.pack(fill=tk.X, pady=2)
        ttk.Label(row, text="Password:", width=12).pack(side=tk.LEFT)
        self.var_pass = tk.StringVar(value="Admin123!")
        ttk.Entry(row, textvariable=self.var_pass, width=18, show='●').pack(side=tk.LEFT, padx=5)

        # Stream type
        row = ttk.Frame(conn_frame)
        row.pack(fill=tk.X, pady=2)
        ttk.Label(row, text="Stream:", width=12).pack(side=tk.LEFT)
        self.var_stream = tk.StringVar(value="main")
        ttk.Radiobutton(row, text="Main (4MP)", variable=self.var_stream, value="main").pack(side=tk.LEFT)
        ttk.Radiobutton(row, text="Sub (720p)", variable=self.var_stream, value="sub").pack(side=tk.LEFT, padx=5)

        # Connect/Disconnect buttons
        btn_row = ttk.Frame(conn_frame)
        btn_row.pack(fill=tk.X, pady=(10, 0))
        self.btn_connect = ttk.Button(btn_row, text="Connect", command=self._on_connect)
        self.btn_connect.pack(side=tk.LEFT, expand=True, fill=tk.X, padx=(0, 5))
        self.btn_disconnect = ttk.Button(btn_row, text="Disconnect", command=self._on_disconnect, state=tk.DISABLED)
        self.btn_disconnect.pack(side=tk.LEFT, expand=True, fill=tk.X)

        # --- PTZ Control Section ---
        ptz_frame = ttk.LabelFrame(ctrl_frame, text="PTZ Control", padding=10)
        ptz_frame.pack(fill=tk.X, pady=(0, 10))

        # Speed slider
        speed_row = ttk.Frame(ptz_frame)
        speed_row.pack(fill=tk.X, pady=(0, 10))
        ttk.Label(speed_row, text="Speed:").pack(side=tk.LEFT)
        self.var_speed = tk.DoubleVar(value=0.5)
        ttk.Scale(speed_row, from_=0.1, to=1.0, variable=self.var_speed,
                  orient=tk.HORIZONTAL, length=150).pack(side=tk.LEFT, padx=10)
        self.lbl_speed = ttk.Label(speed_row, text="0.5", width=4)
        self.lbl_speed.pack(side=tk.LEFT)
        self.var_speed.trace_add('write', lambda *_: self.lbl_speed.config(text=f"{self.var_speed.get():.1f}"))

        # Direction pad
        pad_frame = ttk.Frame(ptz_frame)
        pad_frame.pack()

        btn_cfg = {'width': 5}
        arrows = [
            ('↖', 0, 0, (-1, 1, 0)),   ('↑', 0, 1, (0, 1, 0)),   ('↗', 0, 2, (1, 1, 0)),
            ('←', 1, 0, (-1, 0, 0)),   ('■', 1, 1, None),         ('→', 1, 2, (1, 0, 0)),
            ('↙', 2, 0, (-1, -1, 0)),  ('↓', 2, 1, (0, -1, 0)),  ('↘', 2, 2, (1, -1, 0)),
        ]

        # Configure stop button style
        self.style.configure('Stop.TButton', background='#cc3333', foreground='white')

        for text, row, col, ptz in arrows:
            if ptz is None:
                # Stop button - use the styled button
                btn = ttk.Button(pad_frame, text="STOP", width=5,
                                 style='Stop.TButton', command=self._on_stop)
                btn.grid(row=row, column=col, padx=2, pady=2)
            else:
                btn = ttk.Button(pad_frame, text=text, **btn_cfg)
                btn.grid(row=row, column=col, padx=2, pady=2)
                # Use command for simple click instead of press/release bindings
                btn.configure(command=lambda p=ptz: self._ptz_click(*p))

        # Zoom buttons
        zoom_row = ttk.Frame(ptz_frame)
        zoom_row.pack(fill=tk.X, pady=(10, 0))

        btn_zout = ttk.Button(zoom_row, text="Zoom −", width=10,
                              command=lambda: self._ptz_click(0, 0, -1))
        btn_zout.pack(side=tk.LEFT, expand=True)

        btn_zin = ttk.Button(zoom_row, text="Zoom +", width=10,
                             command=lambda: self._ptz_click(0, 0, 1))
        btn_zin.pack(side=tk.RIGHT, expand=True)

        # --- Absolute Position Section ---
        pos_frame = ttk.LabelFrame(ctrl_frame, text="Absolute Position", padding=10)
        pos_frame.pack(fill=tk.X, pady=(0, 10))

        # Current position display
        cur_row = ttk.Frame(pos_frame)
        cur_row.pack(fill=tk.X, pady=(0, 5))
        ttk.Label(cur_row, text="Current:", width=8).pack(side=tk.LEFT)
        self.lbl_cur_pos = ttk.Label(cur_row, text="Az: --  El: --")
        self.lbl_cur_pos.pack(side=tk.LEFT)
        ttk.Button(cur_row, text="↻", width=2, command=self._update_position).pack(side=tk.RIGHT)

        # Azimuth input
        az_row = ttk.Frame(pos_frame)
        az_row.pack(fill=tk.X, pady=2)
        ttk.Label(az_row, text="Azimuth:", width=8).pack(side=tk.LEFT)
        self.var_azimuth = tk.DoubleVar(value=0.0)
        ttk.Entry(az_row, textvariable=self.var_azimuth, width=8).pack(side=tk.LEFT, padx=5)
        ttk.Label(az_row, text="° (0-360)").pack(side=tk.LEFT)

        # Elevation input
        el_row = ttk.Frame(pos_frame)
        el_row.pack(fill=tk.X, pady=2)
        ttk.Label(el_row, text="Elevation:", width=8).pack(side=tk.LEFT)
        self.var_elevation = tk.DoubleVar(value=0.0)
        ttk.Entry(el_row, textvariable=self.var_elevation, width=8).pack(side=tk.LEFT, padx=5)
        ttk.Label(el_row, text="° (0-90)").pack(side=tk.LEFT)

        # Go To button
        goto_row = ttk.Frame(pos_frame)
        goto_row.pack(fill=tk.X, pady=(5, 0))
        self.btn_goto = ttk.Button(goto_row, text="Go To Position", command=self._on_goto_position)
        self.btn_goto.pack(fill=tk.X)

        # --- Presets Section ---
        preset_frame = ttk.LabelFrame(ctrl_frame, text="Presets", padding=10)
        preset_frame.pack(fill=tk.X, pady=(0, 10))

        # Preset selector
        sel_row = ttk.Frame(preset_frame)
        sel_row.pack(fill=tk.X)
        ttk.Label(sel_row, text="Preset #:").pack(side=tk.LEFT)
        self.var_preset = tk.IntVar(value=1)
        ttk.Spinbox(sel_row, from_=1, to=255, textvariable=self.var_preset, width=5).pack(side=tk.LEFT, padx=5)
        ttk.Button(sel_row, text="Go To", command=self._on_goto_preset, width=8).pack(side=tk.LEFT, padx=5)
        ttk.Button(sel_row, text="Save", command=self._on_save_preset, width=8).pack(side=tk.LEFT)

        # Quick preset buttons 1-8
        quick_row = ttk.Frame(preset_frame)
        quick_row.pack(fill=tk.X, pady=(10, 0))
        for i in range(1, 9):
            btn = ttk.Button(quick_row, text=str(i), width=3, command=lambda p=i: self._on_goto_preset(p))
            btn.pack(side=tk.LEFT, padx=2)

        # --- Keyboard Shortcuts ---
        help_frame = ttk.LabelFrame(ctrl_frame, text="Keyboard Shortcuts", padding=10)
        help_frame.pack(fill=tk.X)

        shortcuts = [
            ("Arrow Keys", "Pan/Tilt"),
            ("+ / −", "Zoom"),
            ("Space / Esc", "Stop"),
            ("1-8", "Presets"),
        ]
        for key, action in shortcuts:
            row = ttk.Frame(help_frame)
            row.pack(fill=tk.X)
            ttk.Label(row, text=key, width=12, font=('TkDefaultFont', 9, 'bold')).pack(side=tk.LEFT)
            ttk.Label(row, text=action).pack(side=tk.LEFT)

    def _create_bindings(self):
        """Set up keyboard bindings."""
        # Arrow keys for PTZ - bind_all ensures it works even when Entry has focus
        self.root.bind_all('<KeyPress-Up>', lambda e: self._on_ptz(0, 1, 0))
        self.root.bind_all('<KeyPress-Down>', lambda e: self._on_ptz(0, -1, 0))
        self.root.bind_all('<KeyPress-Left>', lambda e: self._on_ptz(-1, 0, 0))
        self.root.bind_all('<KeyPress-Right>', lambda e: self._on_ptz(1, 0, 0))

        self.root.bind_all('<KeyRelease-Up>', lambda e: self._on_stop())
        self.root.bind_all('<KeyRelease-Down>', lambda e: self._on_stop())
        self.root.bind_all('<KeyRelease-Left>', lambda e: self._on_stop())
        self.root.bind_all('<KeyRelease-Right>', lambda e: self._on_stop())

        # Zoom
        self.root.bind_all('<KeyPress-plus>', lambda e: self._on_ptz(0, 0, 1))
        self.root.bind_all('<KeyPress-minus>', lambda e: self._on_ptz(0, 0, -1))
        self.root.bind_all('<KeyPress-equal>', lambda e: self._on_ptz(0, 0, 1))
        self.root.bind_all('<KeyRelease-plus>', lambda e: self._on_stop())
        self.root.bind_all('<KeyRelease-minus>', lambda e: self._on_stop())
        self.root.bind_all('<KeyRelease-equal>', lambda e: self._on_stop())

        # Stop
        self.root.bind('<space>', lambda e: self._on_stop())
        self.root.bind('<Escape>', lambda e: self._on_stop())

        # Quick presets
        for i in range(1, 9):
            self.root.bind(f'<Key-{i}>', lambda e, p=i: self._on_goto_preset(p))

        # Window close
        self.root.protocol('WM_DELETE_WINDOW', self._on_quit)

    def _get_rtsp_url(self) -> str:
        """Build RTSP URL from current settings."""
        ip = self.var_ip.get() or "192.168.1.108"
        user = self.var_user.get()
        password = self.var_pass.get()
        subtype = 0 if self.var_stream.get() == "main" else 1
        return EmpireTechPTZ.create_rtsp_url(ip, user, password, 1, subtype)

    def _on_connect(self):
        """Connect to camera."""
        if self._video.is_connected:
            return

        url = self._get_rtsp_url()
        print(f"[GUI] Connecting to: {url}")

        # Create camera instance for PTZ control
        config = CameraConfig(
            name="PTZ Camera",
            rtsp_url=url,
            username=self.var_user.get(),
            password=self.var_pass.get()
        )
        self._camera = EmpireTechPTZ(config)

        # Start video stream
        self._video.start(url)

        self.btn_connect.config(state=tk.DISABLED)
        self.lbl_status.config(text="● Connecting...", foreground='orange')

        # Check connection status
        self.root.after(1500, self._check_connection)

    def _check_connection(self):
        """Check if video connection succeeded."""
        if self._video.is_connected:
            print("[GUI] Connected successfully")
            self.btn_disconnect.config(state=tk.NORMAL)
            self.lbl_status.config(text="● Connected", foreground='green')
        elif self._video.error:
            print(f"[GUI] Connection failed: {self._video.error}")
            self.btn_connect.config(state=tk.NORMAL)
            self.lbl_status.config(text="● Error", foreground='red')
            messagebox.showerror("Connection Error", self._video.error)
        else:
            # Still connecting
            self.root.after(500, self._check_connection)

    def _on_disconnect(self):
        """Disconnect from camera."""
        print("[GUI] Disconnecting")
        self._video.stop()
        self._camera = None

        self.btn_connect.config(state=tk.NORMAL)
        self.btn_disconnect.config(state=tk.DISABLED)
        self.lbl_status.config(text="● Disconnected", foreground='red')
        self.lbl_fps.config(text="FPS: --")
        self.lbl_res.config(text="Resolution: --")

        self.canvas.delete('all')

    def _on_ptz(self, pan: float, tilt: float, zoom: float):
        """Send PTZ movement command."""
        if not self._camera:
            return

        speed = self.var_speed.get()
        print(f"[PTZ] Move: pan={pan}, tilt={tilt}, zoom={zoom}, speed={speed}")
        threading.Thread(
            target=self._camera.ptz_move,
            args=(pan, tilt, zoom, speed),
            daemon=True
        ).start()

    def _ptz_click(self, pan: float, tilt: float, zoom: float):
        """Send PTZ movement for a button click (short burst)."""
        if not self._camera:
            return

        speed = self.var_speed.get()
        print(f"[PTZ] Click: pan={pan}, tilt={tilt}, zoom={zoom}, speed={speed}")

        def move_burst():
            self._camera.ptz_move(pan, tilt, zoom, speed)
            time.sleep(0.3)  # Move for 300ms
            self._camera.ptz_stop()

        threading.Thread(target=move_burst, daemon=True).start()

    def _on_stop(self):
        """Stop PTZ movement."""
        if not self._camera:
            return

        print("[PTZ] Stop")
        threading.Thread(target=self._camera.ptz_stop, daemon=True).start()

    def _on_goto_preset(self, preset: Optional[int] = None):
        """Go to preset position."""
        if not self._camera:
            return

        if preset is None:
            preset = self.var_preset.get()

        print(f"[GUI] Going to preset {preset}")
        threading.Thread(
            target=self._camera.ptz_goto_preset,
            args=(preset,),
            daemon=True
        ).start()

    def _on_save_preset(self):
        """Save current position as preset."""
        if not self._camera:
            return

        preset = self.var_preset.get()
        if messagebox.askyesno("Save Preset", f"Save current position as Preset {preset}?"):
            print(f"[GUI] Saving preset {preset}")
            threading.Thread(
                target=self._camera.ptz_set_preset,
                args=(preset,),
                daemon=True
            ).start()

    def _update_position(self):
        """Update current position display."""
        if not self._camera:
            self.lbl_cur_pos.config(text="Az: --  El: --")
            return

        def fetch_pos():
            pos = self._camera.get_position()
            if pos:
                az, el, zoom = pos
                self.lbl_cur_pos.config(text=f"Az: {az:.1f}°  El: {el:.1f}°")
                # Also update input fields to current position
                self.var_azimuth.set(round(az, 1))
                self.var_elevation.set(round(el, 1))
            else:
                self.lbl_cur_pos.config(text="Az: --  El: --")

        threading.Thread(target=fetch_pos, daemon=True).start()

    def _on_goto_position(self):
        """Move camera to specified azimuth/elevation."""
        if not self._camera:
            return

        az = self.var_azimuth.get()
        el = self.var_elevation.get()

        # Validate inputs
        if not (0 <= az <= 360):
            messagebox.showerror("Invalid Input", "Azimuth must be 0-360 degrees")
            return
        if not (0 <= el <= 90):
            messagebox.showerror("Invalid Input", "Elevation must be 0-90 degrees")
            return

        print(f"[PTZ] Going to Az={az}°, El={el}°")
        self.btn_goto.config(state=tk.DISABLED, text="Moving...")

        def do_goto():
            # goto_position with wait=True will block until camera stops moving
            success = self._camera.goto_position(az, el, wait=True)
            self.btn_goto.config(state=tk.NORMAL, text="Go To Position")
            if success:
                print(f"[PTZ] Reached Az={az}°, El={el}°")
                self._update_position()
            else:
                print("[PTZ] Failed or timed out")

        threading.Thread(target=do_goto, daemon=True).start()

    def _start_video_loop(self):
        """Start the video update loop."""
        self._update_video()

    def _update_video(self):
        """Update video display (called periodically)."""
        if self._video.is_connected:
            frame = self._video.get_frame()

            if frame is not None:
                # Note: Camera has Flip=true set for upside-down mount
                # No software flip needed

                # Update info labels
                w, h = self._video.resolution
                self.lbl_res.config(text=f"Resolution: {w}x{h}")
                self.lbl_fps.config(text=f"FPS: {self._video.fps:.1f}")

                # Get canvas size
                cw = self.canvas.winfo_width()
                ch = self.canvas.winfo_height()

                if cw > 10 and ch > 10:
                    # Scale frame to fit canvas
                    fh, fw = frame.shape[:2]
                    scale = min(cw / fw, ch / fh)
                    new_w = int(fw * scale)
                    new_h = int(fh * scale)

                    frame = cv2.resize(frame, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
                    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

                    # Convert to PhotoImage
                    image = Image.fromarray(frame)
                    photo = ImageTk.PhotoImage(image)

                    # Center on canvas
                    x = (cw - new_w) // 2
                    y = (ch - new_h) // 2

                    self.canvas.delete('all')
                    self.canvas.create_image(x, y, anchor=tk.NW, image=photo)
                    self.canvas._photo = photo  # Keep reference

        # Schedule next update (~30 FPS)
        self._update_job = self.root.after(33, self._update_video)

    def _on_quit(self):
        """Clean up and quit."""
        print("[GUI] Shutting down")
        if self._update_job:
            self.root.after_cancel(self._update_job)
        self._video.stop()
        self.root.destroy()


def main():
    """Main entry point."""
    root = tk.Tk()

    # Set up ttk style
    style = ttk.Style()
    available = style.theme_names()
    if 'clam' in available:
        style.theme_use('clam')
    elif 'alt' in available:
        style.theme_use('alt')

    PTZCameraGUI(root)
    root.mainloop()


if __name__ == "__main__":
    main()
