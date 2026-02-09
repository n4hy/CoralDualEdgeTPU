#!/usr/bin/env python3
"""Run PTZ camera streaming inference benchmark.

Streams video from Empire Tech PTZ425DB-AT camera and runs object detection
on a single Coral Edge TPU, measuring end-to-end latency and throughput.

Usage:
    # Default: 60 second benchmark
    python run_ptz_stream_benchmark.py

    # Custom duration
    python run_ptz_stream_benchmark.py --duration 120

    # Quick test (10 seconds)
    python run_ptz_stream_benchmark.py --quick

    # Custom camera
    python run_ptz_stream_benchmark.py --camera-ip 192.168.1.200

Before running:
1. Ensure TPU permissions are configured (plugdev group)
2. Verify camera is accessible: ping 192.168.1.108
3. Ensure ethernet interface is configured: ip addr show enp97s0
4. Activate the Python environment: source tpuvenv/bin/activate
"""

import argparse
import os
import sys
from datetime import datetime
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

from src.stream_benchmark import PTZStreamBenchmark
from src.dual_tpu import list_edge_tpus


# Default configuration
DEFAULT_MODEL = "models/ssd_mobilenet_v2_coco_quant_postprocess_edgetpu.tflite"
DEFAULT_LABELS = "models/coco_labels.txt"
DEFAULT_CAMERA_IP = "192.168.1.108"
DEFAULT_CAMERA_USER = "admin"
DEFAULT_CAMERA_PASS = os.environ.get("CAMERA_PASS", "")
DEFAULT_DURATION = 60
DEFAULT_WARMUP = 5


def check_prerequisites() -> bool:
    """Verify TPU access."""
    print("Checking prerequisites...")

    # Check TPU devices
    devices = list_edge_tpus()
    if not devices:
        print("\nERROR: No TPU devices found at /dev/apex_*")
        print("Ensure:")
        print("  1. TPU is connected via PCIe")
        print("  2. gasket-dkms and libedgetpu are installed")
        print("  3. User has permissions (plugdev group)")
        return False

    print(f"  Found {len(devices)} TPU device(s): {devices}")
    return True


def check_camera(ip: str) -> bool:
    """Check if camera is reachable."""
    import subprocess
    try:
        result = subprocess.run(
            ["ping", "-c", "1", "-W", "2", ip],
            capture_output=True, timeout=5
        )
        return result.returncode == 0
    except Exception:
        return False


def main():
    parser = argparse.ArgumentParser(
        description="PTZ Camera Streaming Inference Benchmark",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )

    parser.add_argument("--model", "-m", type=str, default=DEFAULT_MODEL,
                       help=f"Path to Edge TPU model (default: {DEFAULT_MODEL})")
    parser.add_argument("--labels", "-l", type=str, default=DEFAULT_LABELS,
                       help=f"Path to labels file (default: {DEFAULT_LABELS})")
    parser.add_argument("--camera-ip", type=str, default=DEFAULT_CAMERA_IP,
                       help=f"Camera IP address (default: {DEFAULT_CAMERA_IP})")
    parser.add_argument("--user", "-u", type=str, default=DEFAULT_CAMERA_USER,
                       help="Camera username")
    parser.add_argument("--password", "-p", type=str,
                       default=DEFAULT_CAMERA_PASS, help="Camera password")
    parser.add_argument("--duration", "-d", type=int, default=DEFAULT_DURATION,
                       help=f"Benchmark duration in seconds (default: {DEFAULT_DURATION})")
    parser.add_argument("--warmup", "-w", type=int, default=DEFAULT_WARMUP,
                       help=f"Warmup duration in seconds (default: {DEFAULT_WARMUP})")
    parser.add_argument("--threshold", "-t", type=float, default=0.5,
                       help="Detection confidence threshold (default: 0.5)")
    parser.add_argument("--device", type=int, default=0,
                       help="TPU device index (default: 0)")
    parser.add_argument("--output-dir", "-o", type=str, default="benchmark_results",
                       help="Output directory for results")
    parser.add_argument("--quick", action="store_true",
                       help="Quick benchmark (10 seconds)")
    parser.add_argument("--skip-camera-check", action="store_true",
                       help="Skip camera connectivity check")

    args = parser.parse_args()

    # Check prerequisites
    if not check_prerequisites():
        return 1

    # Validate model exists
    model_path = Path(args.model)
    if not model_path.exists():
        print(f"\nERROR: Model not found: {model_path}")
        print("\nDownload the model:")
        print("  cd models/")
        print("  wget https://github.com/google-coral/test_data/raw/master/"
              "ssd_mobilenet_v2_coco_quant_postprocess_edgetpu.tflite")
        print("  wget https://raw.githubusercontent.com/google-coral/test_data/"
              "master/coco_labels.txt")
        return 1

    # Check camera connectivity
    if not args.skip_camera_check:
        print(f"  Checking camera at {args.camera_ip}...")
        if not check_camera(args.camera_ip):
            print(f"\nERROR: Cannot reach camera at {args.camera_ip}")
            print("\nEnsure:")
            print("  1. Camera is powered on")
            print("  2. Ethernet interface is configured:")
            print("     sudo ip addr add 192.168.1.100/24 dev enp97s0")
            print("  3. Camera IP is correct (default: 192.168.1.108)")
            print("\nOr use --skip-camera-check to bypass this check")
            return 1
        print("  Camera reachable")

    # Set duration
    duration = 10 if args.quick else args.duration
    warmup = 2 if args.quick else args.warmup

    # Labels path
    labels_path = args.labels if Path(args.labels).exists() else None
    if not labels_path:
        print(f"\nWARNING: Labels file not found: {args.labels}")
        print("  Detections will show class IDs instead of names")

    # Run benchmark
    try:
        benchmark = PTZStreamBenchmark(
            model_path=str(model_path),
            camera_ip=args.camera_ip,
            camera_user=args.user,
            camera_pass=args.password,
            labels_path=labels_path,
            device_idx=args.device,
            detection_threshold=args.threshold
        )

        result = benchmark.run_benchmark(
            duration_sec=duration,
            warmup_sec=warmup
        )

        # Save results
        output_dir = Path(args.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        json_path = output_dir / f"ptz_stream_{timestamp}.json"
        md_path = output_dir / f"ptz_stream_{timestamp}.md"

        benchmark.save_results(str(json_path))
        benchmark.generate_markdown_report(str(md_path))

        print(f"\nResults saved to:")
        print(f"  JSON: {json_path}")
        print(f"  Markdown: {md_path}")

        # Print key metrics for easy copying to README
        print(f"\n{'='*60}")
        print("Key metrics for README:")
        print(f"{'='*60}")
        print(f"Effective FPS: {result.effective_fps:.1f}")
        print(f"Inference latency: {result.inference_latency_mean_ms:.2f}ms mean, "
              f"{result.inference_latency_p99_ms:.2f}ms p99")
        print(f"End-to-end latency: {result.e2e_latency_mean_ms:.2f}ms mean, "
              f"{result.e2e_latency_p99_ms:.2f}ms p99")
        print(f"Frame drop rate: {100*result.frames_dropped/max(1,result.total_frames):.1f}%")

    except KeyboardInterrupt:
        print("\n\nBenchmark interrupted by user")
        return 1
    except Exception as e:
        print(f"\nBenchmark failed: {e}")
        import traceback
        traceback.print_exc()
        return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())
