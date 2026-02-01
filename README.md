# Dual Coral Edge TPU on ARM SBC

High-performance dual Google Coral Edge TPU inference framework for ARM single-board computers. Achieves **216+ inferences/second** with two PCIe TPUs running in parallel.

## Performance Results

Tested on **Orange Pi 6 Plus** (CIX P1 CD8160 ARM SoC) with dual Coral Edge TPU M.2 accelerators.

### Benchmark Summary

| Configuration | Throughput | Latency (mean) | Latency (p99) |
|--------------|------------|----------------|---------------|
| Single TPU | 112.4 inf/s | 8.83 ms | 11.53 ms |
| Dual TPU (parallel) | **216.2 inf/s** | 9.20 ms | 12.68 ms |
| Dual TPU (alternating) | 110.9 inf/s | 8.95 ms | 11.53 ms |

### 5-Minute Stress Test

```
Duration:        300 seconds
Total Inferences: 62,514
Throughput:      205-223 inf/sec (sustained)
Temperature:     43°C (constant, no throttling)
```

### Scaling Efficiency

| Metric | Single TPU | Dual TPU | Scaling |
|--------|-----------|----------|---------|
| Throughput | 112 inf/s | 216 inf/s | **1.93x** |
| Latency p50 | 8.84 ms | 9.06 ms | +2.5% |
| Latency p99 | 11.53 ms | 12.68 ms | +10% |

Near-linear scaling with minimal latency increase demonstrates efficient PCIe bus utilization.

## Hardware

### Tested Configuration

- **SBC**: Orange Pi 6 Plus
- **CPU**: CIX P1 CD8160 (ARM64)
- **TPUs**: 2x Google Coral Edge TPU M.2 A+E key
- **PCIe**: Dual x1 lanes
- **OS**: Linux 6.6.89-cix (aarch64)

### PCIe Topology

```
93:00.0 System peripheral: Global Unichip Corp. Coral Edge TPU
94:00.0 System peripheral: Global Unichip Corp. Coral Edge TPU
```

Both TPUs visible at `/dev/apex_0` and `/dev/apex_1`.

### Supported Cameras

| Camera | Type | Resolution | Features |
|--------|------|------------|----------|
| AXIS M3057-PLVE MK II | Panoramic Dome | 6MP | 360° view, multiple view modes, RTSP |
| Empire Tech PTZ425DB-AT | PTZ | 4MP | 25x zoom, auto-tracking, IR 100m, RTSP |

## Installation

### 1. System Dependencies

```bash
# Add Coral repository
echo "deb https://packages.cloud.google.com/apt coral-edgetpu-stable main" | \
  sudo tee /etc/apt/sources.list.d/coral-edgetpu.list
curl https://packages.cloud.google.com/apt/doc/apt-key.gpg | sudo apt-key add -

# Install Edge TPU runtime and driver
sudo apt update
sudo apt install libedgetpu1-std gasket-dkms
```

### 2. Device Permissions

```bash
# Create udev rule for non-root access
echo 'SUBSYSTEM=="apex", MODE="0660", GROUP="plugdev"' | \
  sudo tee /etc/udev/rules.d/99-apex.rules
sudo udevadm control --reload-rules && sudo udevadm trigger

# Add user to plugdev group (if not already)
sudo usermod -aG plugdev $USER
```

### 3. Python Environment

**Important**: Requires Python 3.9 for Coral-compatible TFLite runtime.

```bash
# Install Python 3.9
pyenv install 3.9.18

# Create virtual environment
~/.pyenv/versions/3.9.18/bin/python -m venv coral39
source coral39/bin/activate

# Install Coral-compatible TFLite runtime
pip install https://github.com/google-coral/pycoral/releases/download/v2.0.0/tflite_runtime-2.5.0.post1-cp39-cp39-linux_aarch64.whl

# Install other dependencies
pip install "numpy<2" "opencv-python-headless<4.10" pillow requests paho-mqtt
```

### 4. Verify Installation

```bash
python -c "
from tflite_runtime.interpreter import Interpreter, load_delegate
d0 = load_delegate('libedgetpu.so.1', {'device': ':0'})
d1 = load_delegate('libedgetpu.so.1', {'device': ':1'})
print('Both TPUs accessible!')
"
```

## Quick Start

### Download Test Model

```bash
cd models/
wget https://github.com/google-coral/test_data/raw/master/ssd_mobilenet_v2_coco_quant_postprocess_edgetpu.tflite
wget https://raw.githubusercontent.com/google-coral/test_data/master/coco_labels.txt
```

### Run Benchmark

```bash
# Activate environment
source coral39/bin/activate

# Quick benchmark (1000 iterations, 60s sustained)
python run_benchmark.py

# Stress test (5000 iterations, 5 min sustained)
python run_benchmark.py --stress
```

### Basic Inference Example

```python
from src import DualEdgeTPU

# Initialize (auto-discovers both TPUs)
tpu = DualEdgeTPU()

# Load model on both TPUs
tpu.load_model("models/ssd_mobilenet_v2_coco_quant_postprocess_edgetpu.tflite")

# Run inference (automatic load balancing)
result = tpu.detect(image_data, threshold=0.5)

# Or specify TPU explicitly
result_tpu0 = tpu.detect(image_data, device_idx=0)
result_tpu1 = tpu.detect(image_data, device_idx=1)
```

## Project Structure

```
CoralDualEdgeTPU/
├── src/
│   ├── dual_tpu.py      # Core dual TPU management
│   ├── camera.py        # AXIS & Empire Tech PTZ camera interface
│   ├── tracker.py       # Object tracking (IoU/centroid)
│   ├── pipeline.py      # Detection + classification pipeline
│   ├── benchmark.py     # Comprehensive benchmark suite
│   └── output.py        # MQTT/webhook publishers
├── examples/
│   ├── basic_inference.py
│   ├── axis_camera_pipeline.py
│   └── sky_watcher.py   # Airplane/satellite detection (dual camera)
├── models/              # Edge TPU compiled models
├── benchmark_results/   # JSON/Markdown benchmark reports
├── coral39/             # Python 3.9 virtual environment
└── run_benchmark.py     # Benchmark runner
```

## Features

### Dual TPU Management

- **Automatic discovery** of all available Edge TPU devices
- **Round-robin load balancing** for single-stream inference
- **Parallel inference** for maximum throughput
- **Thread-safe** operations with per-device locking
- **Independent models** - load different models on each TPU

### Camera Support

- **AXIS M3057-PLVE MK II** panoramic dome camera
- **Empire Tech PTZ425DB-AT** 4MP 25x PTZ camera with tracking
- Generic **RTSP/ONVIF streaming** for any network camera
- **PTZ control** - pan, tilt, zoom, presets
- Multiple **view modes**: panoramic, quad, view areas
- **Auto-reconnect** on network drops
- **Frame buffering** for consistent inference rates

### Object Tracking

- **IoU-based tracker** for crowded scenes
- **Centroid tracker** for sparse objects
- **Track persistence** across frames
- **Velocity estimation** for motion prediction

### Event Publishing

- **MQTT** output with configurable topics
- **HTTP webhooks** with batching and retry
- **Object filtering** by class and confidence
- **Rate limiting** per track

## Benchmark Details

### Test Methodology

Each benchmark run includes:

1. **Warmup**: 50 iterations to reach thermal equilibrium
2. **Timed run**: 1000-5000 iterations with per-inference timing
3. **Thermal monitoring**: CPU temperature sampled every 0.5s
4. **Cooldown**: 5s between tests

### Latency Distribution

```
Single TPU (5000 iterations):
  Mean:   8.83 ms
  Std:    1.00 ms
  Min:    6.53 ms
  Max:   16.21 ms
  p50:    8.84 ms
  p95:   11.11 ms
  p99:   11.53 ms

Dual TPU Parallel (10000 iterations):
  Mean:   9.20 ms
  Std:    1.29 ms
  Min:    6.45 ms
  Max:   14.93 ms
  p50:    9.06 ms
  p95:   11.56 ms
  p99:   12.68 ms
```

### Sustained Load Performance

5-minute continuous inference at maximum throughput:

| Time | Inferences | Rate | Temp |
|------|-----------|------|------|
| 0s | 0 | - | 43°C |
| 60s | 13,198 | 220/s | 43°C |
| 120s | 25,617 | 213/s | 43°C |
| 180s | 37,917 | 211/s | 43°C |
| 240s | 50,336 | 210/s | 43°C |
| 300s | 62,514 | 208/s | 43°C |

**Key findings:**
- No thermal throttling (temperature constant at 43°C)
- Slight throughput decrease over time (~5%) due to system scheduling
- Both TPUs evenly utilized (TPU0: 33,015 / TPU1: 29,499)
- Combined system compute budget:
- AcceleratorTOPS (int8)Status
- CIX Zhouyi V3 NPU~4-6 TOPS*Buggy (single inference only)
- Dual Coral Edge TPU 7.72 TOPS Pending PCIe adapterTotal (working)~ 7.72 TOPS WednesdayTotal (if NPU fixed)~12-14 TOPS Future


## Use Cases

### Sky Watcher (Airplane/Satellite Detection)

```bash
python examples/sky_watcher.py
```

Features:
- **Dual camera support**: AXIS panoramic + Empire Tech PTZ
- Real-time airplane detection with TPU acceleration
- Satellite tracking (solar-illuminated at night)
- PTZ auto-tracking for detected aircraft
- MQTT publishing for external integration
- Object tracking with unique IDs across frames

### Multi-Camera Surveillance

```python
from src import LivePipeline, DualTPUPipeline

pipeline = DualTPUPipeline(
    detection_model="models/ssd_mobilenet_edgetpu.tflite",
    classification_model="models/mobilenet_v2_edgetpu.tflite"
)

live = LivePipeline(pipeline)

# Add AXIS panoramic camera
live.add_axis_camera("panoramic", "192.168.1.100", username="admin", password="pass")

# Add Empire Tech PTZ camera
live.cameras.add_empiretech_ptz("ptz", "192.168.1.101", username="admin", password="pass")

with live:
    for result in live.results():
        for detection in result.detections:
            print(f"[{result.camera_name}] {detection.class_name}: {detection.confidence:.2f}")
```

### PTZ Camera Control

```python
from src import EmpireTechPTZ, CameraConfig

# Create PTZ camera
config = CameraConfig(
    name="ptz-cam",
    rtsp_url=EmpireTechPTZ.create_rtsp_url("192.168.1.101", "admin", "password")
)
ptz = EmpireTechPTZ(config)
ptz.connect()

# Pan right, tilt up
ptz.ptz_move(pan=0.5, tilt=0.3, speed=0.7)
time.sleep(2)
ptz.ptz_stop()

# Zoom to 50%
ptz.zoom_to(0.5)

# Go to preset 1
ptz.ptz_goto_preset(1)

# Save current position as preset 5
ptz.ptz_set_preset(5)
```

## Troubleshooting

### TPU Not Detected

```bash
# Check device nodes
ls -la /dev/apex*

# Check PCIe devices
lspci | grep -i coral

# Check driver loaded
lsmod | grep apex
```

### Permission Denied

```bash
# Verify udev rule
cat /etc/udev/rules.d/99-apex.rules

# Reload rules
sudo udevadm control --reload-rules && sudo udevadm trigger

# Check group membership
groups | grep plugdev
```

### Segmentation Fault

The Edge TPU library requires specific TFLite runtime versions:

- **libedgetpu 16.0** requires **tflite-runtime 2.5.0.post1**
- PyPI's tflite-runtime 2.14.0 is **incompatible**
- Use Python 3.9 with the Coral-provided wheel (see Installation)
- **libedgetpu1-max** causes segfaults even with the correct runtime - use **libedgetpu1-std** only

## Performance Optimization

### Maximum Throughput

For highest aggregate throughput, run both TPUs in parallel:

```python
from concurrent.futures import ThreadPoolExecutor

def infer_on_device(tpu, data, device_idx):
    return tpu.detect(data, device_idx=device_idx)

with ThreadPoolExecutor(max_workers=2) as executor:
    future0 = executor.submit(infer_on_device, tpu, data0, 0)
    future1 = executor.submit(infer_on_device, tpu, data1, 1)
    result0, result1 = future0.result(), future1.result()
```

### Minimum Latency

For lowest single-inference latency, use one TPU:

```python
result = tpu.detect(data, device_idx=0)
```

### Power Management

The standard libedgetpu (`libedgetpu1-std`) uses reduced clock for lower power.

**Note on libedgetpu1-max**: The high-performance library (`libedgetpu1-max`) is currently **incompatible** with this setup:

- `libedgetpu1-max` (v16.0, July 2021) causes segmentation faults when loading models
- The crash occurs during `Interpreter()` initialization with the EdgeTPU delegate
- This affects both `tflite-runtime 2.5.0.post1` (required for libedgetpu 16.0) and newer versions
- Root cause: The max library hasn't been updated since 2021 and has compatibility issues with current TFLite runtimes on aarch64

Until Google releases an updated libedgetpu1-max, use `libedgetpu1-std` which provides stable performance:
- 217+ inferences/second with dual TPUs
- No thermal throttling (41°C sustained)
- Near-linear scaling (1.97x)

## Authors

- **Dr. Robert McGwier, PhD** (N4HY)

## License

MIT License - see [LICENSE](LICENSE) for details.

## Acknowledgments

- [Google Coral](https://coral.ai/) for the Edge TPU hardware and software
- [TensorFlow Lite](https://www.tensorflow.org/lite) for the inference runtime
- Orange Pi for the excellent ARM SBC platform
