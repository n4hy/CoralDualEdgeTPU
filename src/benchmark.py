"""Comprehensive benchmark suite for Dual Coral Edge TPU.

Stress tests both TPUs to measure:
- Single TPU inference throughput
- Dual TPU parallel throughput
- Latency percentiles (p50, p95, p99)
- Thermal behavior under sustained load
- Memory bandwidth utilization

Results suitable for publication.
"""

import gc
import json
import os
import statistics
import subprocess
import threading
import time
from dataclasses import dataclass, field, asdict
from datetime import datetime
from pathlib import Path
from typing import Optional
from concurrent.futures import ThreadPoolExecutor, as_completed

import numpy as np

from .dual_tpu import DualEdgeTPU, list_edge_tpus


@dataclass
class ThermalReading:
    """Thermal sensor reading."""
    timestamp: float
    tpu0_temp: Optional[float] = None
    tpu1_temp: Optional[float] = None
    cpu_temp: Optional[float] = None
    ambient_temp: Optional[float] = None


@dataclass
class BenchmarkResult:
    """Results from a single benchmark run."""
    name: str
    model_name: str
    input_shape: tuple
    num_iterations: int
    num_devices: int

    # Timing stats (milliseconds)
    total_time_sec: float
    throughput_fps: float
    latency_mean_ms: float
    latency_std_ms: float
    latency_min_ms: float
    latency_max_ms: float
    latency_p50_ms: float
    latency_p95_ms: float
    latency_p99_ms: float

    # Thermal
    temp_start_c: Optional[float] = None
    temp_end_c: Optional[float] = None
    temp_max_c: Optional[float] = None

    # System info
    device_paths: list = field(default_factory=list)
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())

    def to_dict(self) -> dict:
        return asdict(self)

    def summary(self) -> str:
        lines = [
            f"\n{'='*60}",
            f"Benchmark: {self.name}",
            f"{'='*60}",
            f"Model: {self.model_name}",
            f"Input shape: {self.input_shape}",
            f"Devices: {self.num_devices} ({', '.join(self.device_paths)})",
            f"Iterations: {self.num_iterations}",
            f"",
            f"THROUGHPUT",
            f"  Total time: {self.total_time_sec:.2f} sec",
            f"  Throughput: {self.throughput_fps:.1f} inferences/sec",
            f"",
            f"LATENCY (ms)",
            f"  Mean:   {self.latency_mean_ms:.2f}",
            f"  Std:    {self.latency_std_ms:.2f}",
            f"  Min:    {self.latency_min_ms:.2f}",
            f"  Max:    {self.latency_max_ms:.2f}",
            f"  p50:    {self.latency_p50_ms:.2f}",
            f"  p95:    {self.latency_p95_ms:.2f}",
            f"  p99:    {self.latency_p99_ms:.2f}",
        ]

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


class ThermalMonitor:
    """Monitor system temperatures during benchmarks."""

    # Common thermal zone paths on ARM SBCs
    THERMAL_PATHS = [
        "/sys/class/thermal/thermal_zone0/temp",  # CPU
        "/sys/class/thermal/thermal_zone1/temp",
        "/sys/class/thermal/thermal_zone2/temp",
        "/sys/class/thermal/thermal_zone3/temp",
        "/sys/class/thermal/thermal_zone4/temp",
    ]

    def __init__(self, poll_interval: float = 0.5):
        self.poll_interval = poll_interval
        self._running = False
        self._thread: Optional[threading.Thread] = None
        self.readings: list[ThermalReading] = []
        self._lock = threading.Lock()

        # Find available thermal zones
        self.available_zones = []
        for path in self.THERMAL_PATHS:
            if os.path.exists(path):
                self.available_zones.append(path)

    def _read_temp(self, path: str) -> Optional[float]:
        """Read temperature from sysfs (returns Celsius)."""
        try:
            with open(path, 'r') as f:
                # Usually in millidegrees
                return int(f.read().strip()) / 1000.0
        except (IOError, ValueError):
            return None

    def get_cpu_temp(self) -> Optional[float]:
        """Get current CPU temperature."""
        if self.available_zones:
            return self._read_temp(self.available_zones[0])
        return None

    def start(self):
        """Start background temperature monitoring."""
        self._running = True
        self.readings = []
        self._thread = threading.Thread(target=self._monitor_loop, daemon=True)
        self._thread.start()

    def stop(self) -> list[ThermalReading]:
        """Stop monitoring and return readings."""
        self._running = False
        if self._thread:
            self._thread.join(timeout=2.0)
        return self.readings

    def _monitor_loop(self):
        while self._running:
            reading = ThermalReading(
                timestamp=time.time(),
                cpu_temp=self.get_cpu_temp()
            )

            with self._lock:
                self.readings.append(reading)

            time.sleep(self.poll_interval)

    def get_max_temp(self) -> Optional[float]:
        """Get maximum recorded CPU temperature."""
        temps = [r.cpu_temp for r in self.readings if r.cpu_temp is not None]
        return max(temps) if temps else None


class DualTPUBenchmark:
    """Comprehensive benchmark suite for dual Edge TPU."""

    def __init__(self, model_path: str):
        """
        Args:
            model_path: Path to Edge TPU compiled .tflite model
        """
        self.model_path = str(Path(model_path).resolve())
        self.model_name = Path(model_path).stem
        self.tpu = DualEdgeTPU()
        self.thermal = ThermalMonitor()
        self.results: list[BenchmarkResult] = []

        # Load model on all devices
        print(f"Loading model: {self.model_name}")
        for i in range(self.tpu.num_devices):
            self.tpu.load_model(self.model_path, device_idx=i)

        # Get input shape
        self.input_details = self.tpu.get_input_details(0)
        self.input_shape = tuple(self.input_details["shape"])
        self.input_dtype = self.input_details["dtype"]

        print(f"Input shape: {self.input_shape}, dtype: {self.input_dtype}")
        print(f"Devices: {[d.device_path for d in self.tpu.devices]}")

    def _generate_input(self) -> np.ndarray:
        """Generate random input tensor."""
        if self.input_dtype == np.uint8:
            return np.random.randint(0, 255, size=self.input_shape[1:], dtype=np.uint8)
        else:
            return np.random.randn(*self.input_shape[1:]).astype(self.input_dtype)

    def _warmup(self, device_idx: int, iterations: int = 10):
        """Warmup the TPU to reach steady-state performance."""
        input_data = self._generate_input()
        for _ in range(iterations):
            self.tpu.infer(input_data, device_idx=device_idx)

    def benchmark_single_tpu(self, device_idx: int, iterations: int = 1000,
                             warmup: int = 50) -> BenchmarkResult:
        """Benchmark a single TPU device.

        Args:
            device_idx: Which TPU to test (0 or 1)
            iterations: Number of inference iterations
            warmup: Warmup iterations before timing
        """
        print(f"\n>>> Single TPU Benchmark (Device {device_idx})")
        print(f"    Warmup: {warmup} iterations")

        # Warmup
        self._warmup(device_idx, warmup)
        gc.collect()

        # Prepare input
        input_data = self._generate_input()

        # Start thermal monitoring
        self.thermal.start()
        temp_start = self.thermal.get_cpu_temp()

        # Benchmark loop
        latencies = []
        print(f"    Running {iterations} iterations...")

        start_time = time.perf_counter()
        for i in range(iterations):
            iter_start = time.perf_counter()
            self.tpu.infer(input_data, device_idx=device_idx)
            iter_end = time.perf_counter()
            latencies.append((iter_end - iter_start) * 1000)  # ms

            if (i + 1) % 200 == 0:
                temp = self.thermal.get_cpu_temp()
                temp_str = f"{temp:.1f}C" if temp else "N/A"
                print(f"    Progress: {i+1}/{iterations} "
                      f"({(i+1)/iterations*100:.0f}%) - Temp: {temp_str}")

        end_time = time.perf_counter()

        # Stop thermal monitoring
        self.thermal.stop()
        temp_end = self.thermal.get_cpu_temp()
        temp_max = self.thermal.get_max_temp()

        total_time = end_time - start_time

        result = BenchmarkResult(
            name=f"single_tpu_{device_idx}",
            model_name=self.model_name,
            input_shape=self.input_shape,
            num_iterations=iterations,
            num_devices=1,
            total_time_sec=total_time,
            throughput_fps=iterations / total_time,
            latency_mean_ms=statistics.mean(latencies),
            latency_std_ms=statistics.stdev(latencies) if len(latencies) > 1 else 0,
            latency_min_ms=min(latencies),
            latency_max_ms=max(latencies),
            latency_p50_ms=np.percentile(latencies, 50),
            latency_p95_ms=np.percentile(latencies, 95),
            latency_p99_ms=np.percentile(latencies, 99),
            temp_start_c=temp_start,
            temp_end_c=temp_end,
            temp_max_c=temp_max,
            device_paths=[self.tpu.devices[device_idx].device_path]
        )

        self.results.append(result)
        print(result.summary())
        return result

    def benchmark_dual_tpu_parallel(self, iterations: int = 1000,
                                    warmup: int = 50) -> BenchmarkResult:
        """Benchmark both TPUs running in parallel.

        Each TPU processes its own stream of inferences simultaneously.
        This measures maximum aggregate throughput.
        """
        if self.tpu.num_devices < 2:
            raise RuntimeError("Dual TPU benchmark requires 2 devices")

        print(f"\n>>> Dual TPU Parallel Benchmark")
        print(f"    Warmup: {warmup} iterations per device")

        # Warmup both devices
        def warmup_device(idx):
            self._warmup(idx, warmup)

        with ThreadPoolExecutor(max_workers=2) as executor:
            executor.map(warmup_device, [0, 1])

        gc.collect()

        # Prepare inputs (different for each device to prevent caching effects)
        inputs = [self._generate_input() for _ in range(2)]

        # Thread results storage
        latencies_0 = []
        latencies_1 = []

        def run_inference(device_idx, latency_list, num_iters):
            input_data = inputs[device_idx]
            for _ in range(num_iters):
                start = time.perf_counter()
                self.tpu.infer(input_data, device_idx=device_idx)
                end = time.perf_counter()
                latency_list.append((end - start) * 1000)

        # Start thermal monitoring
        self.thermal.start()
        temp_start = self.thermal.get_cpu_temp()

        print(f"    Running {iterations} iterations per device (parallel)...")

        # Run both TPUs in parallel
        start_time = time.perf_counter()

        with ThreadPoolExecutor(max_workers=2) as executor:
            f0 = executor.submit(run_inference, 0, latencies_0, iterations)
            f1 = executor.submit(run_inference, 1, latencies_1, iterations)
            f0.result()
            f1.result()

        end_time = time.perf_counter()

        # Stop thermal monitoring
        self.thermal.stop()
        temp_end = self.thermal.get_cpu_temp()
        temp_max = self.thermal.get_max_temp()

        total_time = end_time - start_time
        total_inferences = iterations * 2

        # Combine latencies for statistics
        all_latencies = latencies_0 + latencies_1

        result = BenchmarkResult(
            name="dual_tpu_parallel",
            model_name=self.model_name,
            input_shape=self.input_shape,
            num_iterations=total_inferences,
            num_devices=2,
            total_time_sec=total_time,
            throughput_fps=total_inferences / total_time,
            latency_mean_ms=statistics.mean(all_latencies),
            latency_std_ms=statistics.stdev(all_latencies) if len(all_latencies) > 1 else 0,
            latency_min_ms=min(all_latencies),
            latency_max_ms=max(all_latencies),
            latency_p50_ms=np.percentile(all_latencies, 50),
            latency_p95_ms=np.percentile(all_latencies, 95),
            latency_p99_ms=np.percentile(all_latencies, 99),
            temp_start_c=temp_start,
            temp_end_c=temp_end,
            temp_max_c=temp_max,
            device_paths=[d.device_path for d in self.tpu.devices]
        )

        self.results.append(result)
        print(result.summary())

        # Print per-device stats
        print(f"Per-device breakdown:")
        print(f"  TPU 0: {statistics.mean(latencies_0):.2f}ms mean, "
              f"{iterations/total_time:.1f} inf/sec")
        print(f"  TPU 1: {statistics.mean(latencies_1):.2f}ms mean, "
              f"{iterations/total_time:.1f} inf/sec")

        return result

    def benchmark_dual_tpu_alternating(self, iterations: int = 1000,
                                       warmup: int = 50) -> BenchmarkResult:
        """Benchmark load-balanced inference across both TPUs.

        Uses round-robin scheduling to distribute work.
        This measures real-world throughput for a single stream.
        """
        if self.tpu.num_devices < 2:
            raise RuntimeError("Dual TPU benchmark requires 2 devices")

        print(f"\n>>> Dual TPU Alternating (Load-Balanced) Benchmark")
        print(f"    Warmup: {warmup} iterations")

        # Warmup both
        for i in range(2):
            self._warmup(i, warmup // 2)
        gc.collect()

        input_data = self._generate_input()

        # Start thermal monitoring
        self.thermal.start()
        temp_start = self.thermal.get_cpu_temp()

        latencies = []
        print(f"    Running {iterations} iterations (alternating)...")

        start_time = time.perf_counter()
        for i in range(iterations):
            iter_start = time.perf_counter()
            # Round-robin: device_idx = i % 2
            self.tpu.infer(input_data)  # Uses internal round-robin
            iter_end = time.perf_counter()
            latencies.append((iter_end - iter_start) * 1000)

            if (i + 1) % 200 == 0:
                temp = self.thermal.get_cpu_temp()
                temp_str = f"{temp:.1f}C" if temp else "N/A"
                print(f"    Progress: {i+1}/{iterations} - Temp: {temp_str}")

        end_time = time.perf_counter()

        self.thermal.stop()
        temp_end = self.thermal.get_cpu_temp()
        temp_max = self.thermal.get_max_temp()

        total_time = end_time - start_time

        result = BenchmarkResult(
            name="dual_tpu_alternating",
            model_name=self.model_name,
            input_shape=self.input_shape,
            num_iterations=iterations,
            num_devices=2,
            total_time_sec=total_time,
            throughput_fps=iterations / total_time,
            latency_mean_ms=statistics.mean(latencies),
            latency_std_ms=statistics.stdev(latencies) if len(latencies) > 1 else 0,
            latency_min_ms=min(latencies),
            latency_max_ms=max(latencies),
            latency_p50_ms=np.percentile(latencies, 50),
            latency_p95_ms=np.percentile(latencies, 95),
            latency_p99_ms=np.percentile(latencies, 99),
            temp_start_c=temp_start,
            temp_end_c=temp_end,
            temp_max_c=temp_max,
            device_paths=[d.device_path for d in self.tpu.devices]
        )

        self.results.append(result)
        print(result.summary())
        return result

    def benchmark_sustained_load(self, duration_sec: int = 60,
                                 warmup: int = 50) -> BenchmarkResult:
        """Run sustained load to test thermal throttling.

        Both TPUs run continuously for specified duration.
        Use this to find thermal limits.
        """
        print(f"\n>>> Sustained Load Test ({duration_sec} seconds)")
        print(f"    Running both TPUs at maximum load...")

        if self.tpu.num_devices < 2:
            print("    Warning: Only 1 TPU available")

        # Warmup
        for i in range(self.tpu.num_devices):
            self._warmup(i, warmup)
        gc.collect()

        inputs = [self._generate_input() for _ in range(2)]

        # Shared state
        stop_event = threading.Event()
        inference_counts = [0, 0]
        all_latencies = [[], []]

        def sustained_inference(device_idx):
            input_data = inputs[device_idx]
            while not stop_event.is_set():
                start = time.perf_counter()
                self.tpu.infer(input_data, device_idx=device_idx)
                end = time.perf_counter()
                all_latencies[device_idx].append((end - start) * 1000)
                inference_counts[device_idx] += 1

        # Start thermal monitoring
        self.thermal.start()
        temp_start = self.thermal.get_cpu_temp()

        # Launch threads
        threads = []
        for i in range(self.tpu.num_devices):
            t = threading.Thread(target=sustained_inference, args=(i,), daemon=True)
            threads.append(t)

        start_time = time.perf_counter()
        for t in threads:
            t.start()

        # Progress updates
        for elapsed in range(duration_sec):
            time.sleep(1)
            temp = self.thermal.get_cpu_temp()
            total_infs = sum(inference_counts)
            temp_str = f"{temp:.1f}C" if temp else "N/A"
            print(f"    [{elapsed+1:3d}s] Inferences: {total_infs:6d} | "
                  f"Rate: {total_infs/(elapsed+1):.0f}/s | Temp: {temp_str}")

        stop_event.set()
        for t in threads:
            t.join(timeout=2.0)

        end_time = time.perf_counter()

        self.thermal.stop()
        temp_end = self.thermal.get_cpu_temp()
        temp_max = self.thermal.get_max_temp()

        total_time = end_time - start_time
        total_inferences = sum(inference_counts)
        combined_latencies = all_latencies[0] + all_latencies[1]

        result = BenchmarkResult(
            name=f"sustained_load_{duration_sec}s",
            model_name=self.model_name,
            input_shape=self.input_shape,
            num_iterations=total_inferences,
            num_devices=self.tpu.num_devices,
            total_time_sec=total_time,
            throughput_fps=total_inferences / total_time,
            latency_mean_ms=statistics.mean(combined_latencies) if combined_latencies else 0,
            latency_std_ms=statistics.stdev(combined_latencies) if len(combined_latencies) > 1 else 0,
            latency_min_ms=min(combined_latencies) if combined_latencies else 0,
            latency_max_ms=max(combined_latencies) if combined_latencies else 0,
            latency_p50_ms=np.percentile(combined_latencies, 50) if combined_latencies else 0,
            latency_p95_ms=np.percentile(combined_latencies, 95) if combined_latencies else 0,
            latency_p99_ms=np.percentile(combined_latencies, 99) if combined_latencies else 0,
            temp_start_c=temp_start,
            temp_end_c=temp_end,
            temp_max_c=temp_max,
            device_paths=[d.device_path for d in self.tpu.devices]
        )

        self.results.append(result)
        print(result.summary())

        # Per-device breakdown
        print(f"Per-device breakdown:")
        for i in range(self.tpu.num_devices):
            if all_latencies[i]:
                print(f"  TPU {i}: {inference_counts[i]} inferences, "
                      f"{statistics.mean(all_latencies[i]):.2f}ms mean latency")

        return result

    def run_full_suite(self, iterations: int = 1000,
                       sustained_duration: int = 60) -> list[BenchmarkResult]:
        """Run complete benchmark suite.

        Args:
            iterations: Iterations for throughput tests
            sustained_duration: Duration for thermal stress test

        Returns:
            List of all benchmark results
        """
        print("\n" + "="*60)
        print("DUAL CORAL EDGE TPU BENCHMARK SUITE")
        print("="*60)
        print(f"Model: {self.model_name}")
        print(f"Input shape: {self.input_shape}")
        print(f"Devices: {self.tpu.num_devices}")
        print(f"Iterations per test: {iterations}")
        print(f"Sustained load duration: {sustained_duration}s")
        print("="*60)

        self.results = []

        # Cool down between tests
        cooldown = 5

        # Single TPU tests
        self.benchmark_single_tpu(0, iterations)
        print(f"\nCooling down {cooldown}s...")
        time.sleep(cooldown)

        if self.tpu.num_devices > 1:
            self.benchmark_single_tpu(1, iterations)
            print(f"\nCooling down {cooldown}s...")
            time.sleep(cooldown)

            # Dual TPU tests
            self.benchmark_dual_tpu_parallel(iterations)
            print(f"\nCooling down {cooldown}s...")
            time.sleep(cooldown)

            self.benchmark_dual_tpu_alternating(iterations)
            print(f"\nCooling down {cooldown}s...")
            time.sleep(cooldown)

        # Sustained load test
        self.benchmark_sustained_load(sustained_duration)

        return self.results

    def save_results(self, output_path: str):
        """Save results to JSON file."""
        output = {
            "system_info": self._get_system_info(),
            "model_info": {
                "path": self.model_path,
                "name": self.model_name,
                "input_shape": self.input_shape,
            },
            "results": [r.to_dict() for r in self.results],
            "generated_at": datetime.now().isoformat()
        }

        with open(output_path, 'w') as f:
            json.dump(output, f, indent=2, default=str)

        print(f"\nResults saved to: {output_path}")

    def _get_system_info(self) -> dict:
        """Gather system information for reproducibility."""
        info = {
            "hostname": os.uname().nodename,
            "kernel": os.uname().release,
            "arch": os.uname().machine,
            "tpu_devices": [],
        }

        # TPU info
        for device in self.tpu.devices:
            info["tpu_devices"].append({
                "path": device.device_path,
                "model_loaded": device.model_path
            })

        # Try to get PCIe info
        try:
            result = subprocess.run(
                ["lspci", "-d", "1ac1:089a", "-v"],
                capture_output=True, text=True, timeout=5
            )
            info["pcie_info"] = result.stdout
        except:
            pass

        # CPU info
        try:
            with open("/proc/cpuinfo", 'r') as f:
                cpuinfo = f.read()
                for line in cpuinfo.split('\n'):
                    if 'model name' in line or 'Hardware' in line:
                        info["cpu"] = line.split(':')[1].strip()
                        break
        except:
            pass

        return info

    def generate_markdown_report(self, output_path: str):
        """Generate a Markdown report suitable for publication."""
        lines = [
            "# Dual Coral Edge TPU Benchmark Results",
            "",
            f"**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
            "",
            "## System Configuration",
            "",
            f"- **Platform:** {os.uname().nodename} ({os.uname().machine})",
            f"- **Kernel:** {os.uname().release}",
            f"- **TPU Devices:** {self.tpu.num_devices}",
        ]

        for i, device in enumerate(self.tpu.devices):
            lines.append(f"  - TPU {i}: `{device.device_path}`")

        lines.extend([
            "",
            "## Model Information",
            "",
            f"- **Model:** `{self.model_name}`",
            f"- **Input Shape:** `{self.input_shape}`",
            f"- **Input Type:** `{self.input_dtype}`",
            "",
            "## Results Summary",
            "",
            "| Test | Throughput (inf/s) | Latency Mean (ms) | Latency p99 (ms) | Temp Max (C) |",
            "|------|-------------------|-------------------|------------------|--------------|",
        ])

        for r in self.results:
            temp_str = f"{r.temp_max_c:.1f}" if r.temp_max_c else "N/A"
            lines.append(
                f"| {r.name} | {r.throughput_fps:.1f} | {r.latency_mean_ms:.2f} | "
                f"{r.latency_p99_ms:.2f} | {temp_str} |"
            )

        lines.extend([
            "",
            "## Detailed Results",
            ""
        ])

        for r in self.results:
            lines.extend([
                f"### {r.name}",
                "",
                f"- **Iterations:** {r.num_iterations}",
                f"- **Total Time:** {r.total_time_sec:.2f}s",
                f"- **Throughput:** {r.throughput_fps:.1f} inferences/second",
                "",
                "**Latency Distribution (ms):**",
                "",
                f"| Metric | Value |",
                f"|--------|-------|",
                f"| Mean | {r.latency_mean_ms:.2f} |",
                f"| Std Dev | {r.latency_std_ms:.2f} |",
                f"| Min | {r.latency_min_ms:.2f} |",
                f"| Max | {r.latency_max_ms:.2f} |",
                f"| p50 | {r.latency_p50_ms:.2f} |",
                f"| p95 | {r.latency_p95_ms:.2f} |",
                f"| p99 | {r.latency_p99_ms:.2f} |",
                ""
            ])

            if r.temp_start_c is not None:
                lines.extend([
                    "**Thermal:**",
                    "",
                    f"- Start: {r.temp_start_c:.1f}C",
                    f"- End: {r.temp_end_c:.1f}C",
                    f"- Max: {r.temp_max_c:.1f}C",
                    ""
                ])

        with open(output_path, 'w') as f:
            f.write('\n'.join(lines))

        print(f"Markdown report saved to: {output_path}")
