"""PTZ camera streaming inference benchmark.

Measures real-time inference performance on live video stream:
- Frame capture latency
- Preprocessing latency
- TPU inference latency
- End-to-end latency
- Detection statistics
- Thermal behavior
"""

import gc
import json
import os
import statistics
import time
from dataclasses import dataclass, field, asdict
from datetime import datetime
from pathlib import Path
from typing import Optional

import cv2
import numpy as np

from .dual_tpu import DualEdgeTPU
from .camera import EmpireTechPTZ, CameraConfig
from .benchmark import ThermalMonitor


@dataclass
class FrameMetrics:
    """Metrics for a single processed frame."""
    frame_number: int
    capture_timestamp: float
    preprocess_start: float
    preprocess_end: float
    inference_start: float
    inference_end: float
    num_detections: int
    detection_classes: list[int] = field(default_factory=list)

    @property
    def capture_latency_ms(self) -> float:
        """Time from frame timestamp to preprocess start."""
        return (self.preprocess_start - self.capture_timestamp) * 1000

    @property
    def preprocess_latency_ms(self) -> float:
        """Preprocessing time (resize + color conversion)."""
        return (self.preprocess_end - self.preprocess_start) * 1000

    @property
    def inference_latency_ms(self) -> float:
        """TPU inference time."""
        return (self.inference_end - self.inference_start) * 1000

    @property
    def e2e_latency_ms(self) -> float:
        """End-to-end latency from capture to result."""
        return (self.inference_end - self.capture_timestamp) * 1000


@dataclass
class StreamBenchmarkResult:
    """Results from PTZ camera streaming inference benchmark."""
    name: str
    model_name: str
    camera_name: str
    camera_resolution: tuple[int, int]
    input_shape: tuple

    # Duration
    duration_sec: float

    # Frame statistics
    total_frames: int
    frames_processed: int
    frames_dropped: int

    # Throughput
    capture_fps: float
    inference_fps: float
    effective_fps: float

    # Capture latency (ms)
    capture_latency_mean_ms: float
    capture_latency_std_ms: float
    capture_latency_p95_ms: float
    capture_latency_p99_ms: float

    # Preprocess latency (ms)
    preprocess_latency_mean_ms: float
    preprocess_latency_std_ms: float

    # Inference latency (ms)
    inference_latency_mean_ms: float
    inference_latency_std_ms: float
    inference_latency_min_ms: float
    inference_latency_max_ms: float
    inference_latency_p50_ms: float
    inference_latency_p95_ms: float
    inference_latency_p99_ms: float

    # End-to-end latency (ms)
    e2e_latency_mean_ms: float
    e2e_latency_std_ms: float
    e2e_latency_p50_ms: float
    e2e_latency_p95_ms: float
    e2e_latency_p99_ms: float

    # Detection statistics
    total_detections: int
    detections_per_frame_mean: float
    detections_per_class: dict[str, int] = field(default_factory=dict)

    # Thermal
    temp_start_c: Optional[float] = None
    temp_end_c: Optional[float] = None
    temp_max_c: Optional[float] = None

    # System info
    device_path: str = ""
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())

    def to_dict(self) -> dict:
        return asdict(self)

    def summary(self) -> str:
        """Generate human-readable summary."""
        lines = [
            f"\n{'='*60}",
            f"PTZ Streaming Benchmark Results",
            f"{'='*60}",
            f"Model: {self.model_name}",
            f"Camera: {self.camera_name} ({self.camera_resolution[0]}x{self.camera_resolution[1]})",
            f"TPU Device: {self.device_path}",
            f"Duration: {self.duration_sec:.1f}s",
            f"",
            f"FRAME STATISTICS",
            f"  Frames processed: {self.frames_processed}",
            f"  Frames dropped:   {self.frames_dropped} ({100*self.frames_dropped/max(1,self.total_frames):.1f}%)",
            f"  Effective FPS:    {self.effective_fps:.1f}",
            f"",
            f"LATENCY BREAKDOWN (ms)",
            f"  Capture:    {self.capture_latency_mean_ms:.2f} mean, {self.capture_latency_p99_ms:.2f} p99",
            f"  Preprocess: {self.preprocess_latency_mean_ms:.2f} mean",
            f"  Inference:  {self.inference_latency_mean_ms:.2f} mean, {self.inference_latency_p99_ms:.2f} p99",
            f"  End-to-End: {self.e2e_latency_mean_ms:.2f} mean, {self.e2e_latency_p99_ms:.2f} p99",
            f"",
            f"INFERENCE LATENCY DETAIL (ms)",
            f"  Mean:   {self.inference_latency_mean_ms:.2f}",
            f"  Std:    {self.inference_latency_std_ms:.2f}",
            f"  Min:    {self.inference_latency_min_ms:.2f}",
            f"  Max:    {self.inference_latency_max_ms:.2f}",
            f"  p50:    {self.inference_latency_p50_ms:.2f}",
            f"  p95:    {self.inference_latency_p95_ms:.2f}",
            f"  p99:    {self.inference_latency_p99_ms:.2f}",
            f"",
            f"DETECTIONS",
            f"  Total:     {self.total_detections}",
            f"  Per frame: {self.detections_per_frame_mean:.2f}",
        ]

        if self.detections_per_class:
            lines.append(f"  By class:")
            sorted_classes = sorted(self.detections_per_class.items(),
                                    key=lambda x: x[1], reverse=True)
            for cls_name, count in sorted_classes[:5]:
                lines.append(f"    {cls_name}: {count}")

        if self.temp_start_c is not None:
            lines.extend([
                f"",
                f"THERMAL",
                f"  Start: {self.temp_start_c:.1f}C",
                f"  End:   {self.temp_end_c:.1f}C",
                f"  Max:   {self.temp_max_c:.1f}C",
                f"  Delta: {(self.temp_end_c - self.temp_start_c):+.1f}C",
            ])

        lines.append(f"{'='*60}\n")
        return "\n".join(lines)


class PTZStreamBenchmark:
    """Benchmark inference performance on live PTZ camera stream."""

    def __init__(self,
                 model_path: str,
                 camera_ip: str = "192.168.1.108",
                 camera_user: str = "admin",
                 camera_pass: str = "Admin123!",
                 labels_path: Optional[str] = None,
                 device_idx: int = 0,
                 detection_threshold: float = 0.5):
        """
        Args:
            model_path: Path to Edge TPU compiled detection model
            camera_ip: PTZ camera IP address
            camera_user: Camera username
            camera_pass: Camera password
            labels_path: Path to COCO labels file
            device_idx: TPU device index (0 for single TPU mode)
            detection_threshold: Minimum detection confidence
        """
        self.model_path = str(Path(model_path).resolve())
        self.model_name = Path(model_path).stem
        self.device_idx = device_idx
        self.detection_threshold = detection_threshold
        self.camera_ip = camera_ip

        # Load labels
        self.labels = self._load_labels(labels_path) if labels_path else {}

        # Initialize TPU
        print(f"Initializing TPU device {device_idx}...")
        self.tpu = DualEdgeTPU()
        self.tpu.load_model(self.model_path, device_idx=self.device_idx)

        # Get model input shape
        self.input_details = self.tpu.get_input_details(self.device_idx)
        self.input_shape = tuple(self.input_details["shape"])
        self.input_size = (self.input_shape[1], self.input_shape[2])  # (H, W)

        # Initialize camera
        self.camera = self._create_camera(camera_ip, camera_user, camera_pass)

        # Thermal monitoring
        self.thermal = ThermalMonitor()

        # Results storage
        self.frame_metrics: list[FrameMetrics] = []
        self.result: Optional[StreamBenchmarkResult] = None

    def _create_camera(self, ip: str, user: str, password: str) -> EmpireTechPTZ:
        """Create PTZ camera instance."""
        # URL already includes credentials, so don't pass them to CameraConfig
        # (get_auth_url would add them again causing auth failure)
        url = EmpireTechPTZ.create_rtsp_url(ip, user, password, subtype=0)
        config = CameraConfig(
            name="ptz-benchmark",
            rtsp_url=url,
            resolution=(2560, 1440),
            buffer_size=2
        )
        return EmpireTechPTZ(config)

    def _load_labels(self, path: str) -> dict[int, str]:
        """Load labels file (one label per line)."""
        labels = {}
        with open(path, 'r') as f:
            for i, line in enumerate(f):
                labels[i] = line.strip()
        return labels

    def preprocess(self, image: np.ndarray) -> np.ndarray:
        """Preprocess frame for inference."""
        resized = cv2.resize(image, (self.input_size[1], self.input_size[0]))
        rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
        return rgb

    def run_benchmark(self, duration_sec: int = 60,
                      warmup_sec: int = 5) -> StreamBenchmarkResult:
        """Run the streaming benchmark.

        Args:
            duration_sec: Benchmark duration in seconds
            warmup_sec: Warmup period before timing starts

        Returns:
            StreamBenchmarkResult with all metrics
        """
        print(f"\n{'='*60}")
        print(f"PTZ Camera Streaming Benchmark")
        print(f"{'='*60}")
        print(f"Model: {self.model_name}")
        print(f"Input shape: {self.input_shape}")
        print(f"Camera: {self.camera_ip} (2560x1440)")
        print(f"TPU Device: /dev/apex_{self.device_idx}")
        print(f"Duration: {duration_sec}s (warmup: {warmup_sec}s)")
        print(f"{'='*60}\n")

        # Connect to camera
        print("Connecting to camera...")
        if not self.camera.connect():
            raise RuntimeError("Failed to connect to camera")

        self.camera.start()

        # Warmup phase
        print(f"Warmup phase ({warmup_sec}s)...")
        warmup_end = time.time() + warmup_sec
        warmup_count = 0
        while time.time() < warmup_end:
            frame = self.camera.get_frame(timeout=1.0)
            if frame:
                input_data = self.preprocess(frame.image)
                self.tpu.detect(input_data, device_idx=self.device_idx,
                               threshold=self.detection_threshold)
                warmup_count += 1

        print(f"  Warmup complete: {warmup_count} frames")
        gc.collect()

        # Clear metrics
        self.frame_metrics = []

        # Start thermal monitoring
        self.thermal.start()
        temp_start = self.thermal.get_cpu_temp()

        # Benchmark loop
        print(f"Running benchmark for {duration_sec}s...")
        start_time = time.time()
        end_time = start_time + duration_sec
        frame_count = 0
        dropped_frames = 0
        last_progress = 0

        while time.time() < end_time:
            frame = self.camera.get_frame(timeout=0.5)

            if frame is None:
                dropped_frames += 1
                continue

            # Use perf_counter for all timing (frame.timestamp uses time.time)
            capture_ts = time.perf_counter()
            frame_count += 1

            # Preprocess
            preprocess_start = time.perf_counter()
            input_data = self.preprocess(frame.image)
            preprocess_end = time.perf_counter()

            # Inference
            inference_start = time.perf_counter()
            detections = self.tpu.detect(input_data, device_idx=self.device_idx,
                                         threshold=self.detection_threshold)
            inference_end = time.perf_counter()

            # Record metrics
            metrics = FrameMetrics(
                frame_number=frame_count,
                capture_timestamp=capture_ts,
                preprocess_start=preprocess_start,
                preprocess_end=preprocess_end,
                inference_start=inference_start,
                inference_end=inference_end,
                num_detections=len(detections),
                detection_classes=[d.id for d in detections]
            )
            self.frame_metrics.append(metrics)

            # Progress update every 10 seconds
            elapsed = time.time() - start_time
            if int(elapsed / 10) > last_progress:
                last_progress = int(elapsed / 10)
                temp = self.thermal.get_cpu_temp()
                fps = frame_count / elapsed
                temp_str = f"{temp:.1f}C" if temp else "N/A"
                print(f"  [{elapsed:.0f}s] Frames: {frame_count} | "
                      f"FPS: {fps:.1f} | Detections: {len(detections)} | "
                      f"Temp: {temp_str}")

        # Stop thermal monitoring
        self.thermal.stop()
        temp_end = self.thermal.get_cpu_temp()
        temp_max = self.thermal.get_max_temp()

        # Stop camera
        self.camera.stop()
        self.camera.disconnect()

        # Calculate results
        actual_duration = time.time() - start_time
        self.result = self._calculate_results(
            actual_duration, frame_count, dropped_frames,
            temp_start, temp_end, temp_max
        )

        print(self.result.summary())
        return self.result

    def _calculate_results(self, duration: float, frames: int, dropped: int,
                          temp_start: Optional[float], temp_end: Optional[float],
                          temp_max: Optional[float]) -> StreamBenchmarkResult:
        """Calculate benchmark results from collected metrics."""
        if not self.frame_metrics:
            raise RuntimeError("No frames processed")

        # Extract timing arrays
        capture_latencies = [m.capture_latency_ms for m in self.frame_metrics]
        preprocess_latencies = [m.preprocess_latency_ms for m in self.frame_metrics]
        inference_latencies = [m.inference_latency_ms for m in self.frame_metrics]
        e2e_latencies = [m.e2e_latency_ms for m in self.frame_metrics]

        # Detection counts by class
        detection_counts: dict[str, int] = {}
        total_detections = 0
        for m in self.frame_metrics:
            total_detections += m.num_detections
            for class_id in m.detection_classes:
                class_name = self.labels.get(class_id, str(class_id))
                detection_counts[class_name] = detection_counts.get(class_name, 0) + 1

        return StreamBenchmarkResult(
            name="ptz_stream_benchmark",
            model_name=self.model_name,
            camera_name="Empire Tech PTZ425DB-AT",
            camera_resolution=(2560, 1440),
            input_shape=self.input_shape,
            duration_sec=duration,
            total_frames=frames + dropped,
            frames_processed=frames,
            frames_dropped=dropped,
            capture_fps=(frames + dropped) / duration if duration > 0 else 0,
            inference_fps=frames / duration if duration > 0 else 0,
            effective_fps=frames / duration if duration > 0 else 0,

            # Capture latency stats
            capture_latency_mean_ms=statistics.mean(capture_latencies),
            capture_latency_std_ms=statistics.stdev(capture_latencies) if len(capture_latencies) > 1 else 0,
            capture_latency_p95_ms=float(np.percentile(capture_latencies, 95)),
            capture_latency_p99_ms=float(np.percentile(capture_latencies, 99)),

            # Preprocess latency stats
            preprocess_latency_mean_ms=statistics.mean(preprocess_latencies),
            preprocess_latency_std_ms=statistics.stdev(preprocess_latencies) if len(preprocess_latencies) > 1 else 0,

            # Inference latency stats
            inference_latency_mean_ms=statistics.mean(inference_latencies),
            inference_latency_std_ms=statistics.stdev(inference_latencies) if len(inference_latencies) > 1 else 0,
            inference_latency_min_ms=min(inference_latencies),
            inference_latency_max_ms=max(inference_latencies),
            inference_latency_p50_ms=float(np.percentile(inference_latencies, 50)),
            inference_latency_p95_ms=float(np.percentile(inference_latencies, 95)),
            inference_latency_p99_ms=float(np.percentile(inference_latencies, 99)),

            # End-to-end latency stats
            e2e_latency_mean_ms=statistics.mean(e2e_latencies),
            e2e_latency_std_ms=statistics.stdev(e2e_latencies) if len(e2e_latencies) > 1 else 0,
            e2e_latency_p50_ms=float(np.percentile(e2e_latencies, 50)),
            e2e_latency_p95_ms=float(np.percentile(e2e_latencies, 95)),
            e2e_latency_p99_ms=float(np.percentile(e2e_latencies, 99)),

            # Detection stats
            total_detections=total_detections,
            detections_per_frame_mean=total_detections / frames if frames > 0 else 0,
            detections_per_class=detection_counts,

            # Thermal
            temp_start_c=temp_start,
            temp_end_c=temp_end,
            temp_max_c=temp_max,

            device_path=f"/dev/apex_{self.device_idx}"
        )

    def save_results(self, output_path: str):
        """Save results to JSON file."""
        if not self.result:
            raise RuntimeError("No results to save. Run benchmark first.")

        output = {
            "system_info": self._get_system_info(),
            "camera_info": {
                "name": "Empire Tech PTZ425DB-AT",
                "ip": self.camera_ip,
                "resolution": "2560x1440",
                "stream": "Main (4MP)",
                "protocol": "RTSP"
            },
            "model_info": {
                "path": self.model_path,
                "name": self.model_name,
                "input_shape": self.input_shape,
            },
            "result": self.result.to_dict(),
            "generated_at": datetime.now().isoformat()
        }

        with open(output_path, 'w') as f:
            json.dump(output, f, indent=2, default=str)

        print(f"Results saved to: {output_path}")

    def generate_markdown_report(self, output_path: str):
        """Generate Markdown report suitable for README."""
        if not self.result:
            raise RuntimeError("No results to generate report. Run benchmark first.")

        r = self.result
        lines = [
            "# PTZ Camera Streaming Inference Benchmark",
            "",
            f"**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
            "",
            "## Configuration",
            "",
            "### Camera",
            f"- **Model:** {r.camera_name}",
            f"- **Resolution:** {r.camera_resolution[0]}x{r.camera_resolution[1]} (4MP)",
            "- **Stream:** Main stream (RTSP)",
            f"- **IP:** {self.camera_ip}",
            "",
            "### TPU",
            f"- **Device:** `{r.device_path}`",
            f"- **Model:** `{r.model_name}`",
            f"- **Input Shape:** `{r.input_shape}`",
            f"- **Detection Threshold:** {self.detection_threshold}",
            "",
            "## Results Summary",
            "",
            "| Metric | Value |",
            "|--------|-------|",
            f"| Duration | {r.duration_sec:.1f} sec |",
            f"| Frames Processed | {r.frames_processed} |",
            f"| Frames Dropped | {r.frames_dropped} ({100*r.frames_dropped/max(1,r.total_frames):.1f}%) |",
            f"| Effective FPS | {r.effective_fps:.1f} |",
            f"| Total Detections | {r.total_detections} |",
            f"| Detections/Frame | {r.detections_per_frame_mean:.2f} |",
            "",
            "## Latency Breakdown",
            "",
            "### End-to-End Latency (capture to result)",
            "",
            "| Metric | Value (ms) |",
            "|--------|------------|",
            f"| Mean | {r.e2e_latency_mean_ms:.2f} |",
            f"| Std Dev | {r.e2e_latency_std_ms:.2f} |",
            f"| p50 | {r.e2e_latency_p50_ms:.2f} |",
            f"| p95 | {r.e2e_latency_p95_ms:.2f} |",
            f"| p99 | {r.e2e_latency_p99_ms:.2f} |",
            "",
            "### Inference Latency (TPU only)",
            "",
            "| Metric | Value (ms) |",
            "|--------|------------|",
            f"| Mean | {r.inference_latency_mean_ms:.2f} |",
            f"| Std Dev | {r.inference_latency_std_ms:.2f} |",
            f"| Min | {r.inference_latency_min_ms:.2f} |",
            f"| Max | {r.inference_latency_max_ms:.2f} |",
            f"| p50 | {r.inference_latency_p50_ms:.2f} |",
            f"| p95 | {r.inference_latency_p95_ms:.2f} |",
            f"| p99 | {r.inference_latency_p99_ms:.2f} |",
            "",
            "### Preprocessing Latency (resize + color conversion)",
            "",
            "| Metric | Value (ms) |",
            "|--------|------------|",
            f"| Mean | {r.preprocess_latency_mean_ms:.2f} |",
            f"| Std Dev | {r.preprocess_latency_std_ms:.2f} |",
            "",
        ]

        if r.detections_per_class:
            lines.extend([
                "## Detection Statistics",
                "",
                "| Class | Count | % of Total |",
                "|-------|-------|------------|",
            ])
            sorted_classes = sorted(r.detections_per_class.items(),
                                    key=lambda x: x[1], reverse=True)
            for cls_name, count in sorted_classes[:10]:
                pct = 100 * count / r.total_detections if r.total_detections > 0 else 0
                lines.append(f"| {cls_name} | {count} | {pct:.1f}% |")
            lines.append("")

        if r.temp_start_c is not None:
            lines.extend([
                "## Thermal",
                "",
                "| Metric | Value |",
                "|--------|-------|",
                f"| Start | {r.temp_start_c:.1f} C |",
                f"| End | {r.temp_end_c:.1f} C |",
                f"| Max | {r.temp_max_c:.1f} C |",
                f"| Delta | {(r.temp_end_c - r.temp_start_c):+.1f} C |",
                "",
            ])

        with open(output_path, 'w') as f:
            f.write('\n'.join(lines))

        print(f"Markdown report saved to: {output_path}")

    def _get_system_info(self) -> dict:
        """Gather system information for reproducibility."""
        info = {
            "hostname": os.uname().nodename,
            "kernel": os.uname().release,
            "arch": os.uname().machine,
            "tpu_device": {
                "path": f"/dev/apex_{self.device_idx}",
                "model_loaded": self.model_path
            },
        }

        # CPU info
        try:
            with open("/proc/cpuinfo", 'r') as f:
                cpuinfo = f.read()
                for line in cpuinfo.split('\n'):
                    if 'model name' in line or 'Hardware' in line:
                        info["cpu"] = line.split(':')[1].strip()
                        break
        except Exception:
            pass

        return info
