#!/usr/bin/env python3
"""Run comprehensive Dual Coral Edge TPU benchmarks.

This script stress-tests both TPU devices and generates publishable results.

Usage:
    # Quick benchmark (default model, 1000 iterations)
    python run_benchmark.py

    # Custom model
    python run_benchmark.py --model models/efficientdet_lite0_edgetpu.tflite

    # Full stress test (5 minute sustained load)
    python run_benchmark.py --iterations 5000 --sustained 300

    # Output to specific directory
    python run_benchmark.py --output-dir results/

Before running:
1. Fix TPU permissions:
   echo 'SUBSYSTEM=="apex", MODE="0660", GROUP="plugdev"' | sudo tee /etc/udev/rules.d/99-apex.rules
   sudo udevadm control --reload-rules && sudo udevadm trigger

2. Download a model:
   cd models/
   wget https://github.com/google-coral/test_data/raw/master/ssd_mobilenet_v2_coco_quant_postprocess_edgetpu.tflite
"""

import argparse
import os
import sys
from datetime import datetime
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent))

from src.benchmark import DualTPUBenchmark
from src.dual_tpu import check_tpu_status


# Default models to benchmark
DEFAULT_MODELS = [
    "ssd_mobilenet_v2_coco_quant_postprocess_edgetpu.tflite",
    "efficientdet_lite0_320_ptq_edgetpu.tflite",
    "mobilenet_v2_1.0_224_quant_edgetpu.tflite",
]


def check_prerequisites():
    """Verify TPU access before benchmarking."""
    print("Checking TPU status...")
    status = check_tpu_status()

    if not status["devices"]:
        print("\nERROR: No TPU devices found at /dev/apex_*")
        print("Check that the Coral PCIe driver is loaded: lsmod | grep apex")
        return False

    if not status["driver_loaded"]:
        print("\nWARNING: apex driver not detected in lsmod")

    print(f"Found {len(status['devices'])} TPU device(s)")
    for dev in status["devices"]:
        print(f"  {dev['path']}")

    # Check PCIe
    if status["pcie_devices"]:
        print(f"PCIe devices:")
        for pcie in status["pcie_devices"]:
            print(f"  {pcie}")

    # Test access
    for dev in status["devices"]:
        path = dev["path"]
        if not os.access(path, os.R_OK | os.W_OK):
            print(f"\nERROR: Cannot access {path}")
            print("Fix permissions:")
            print("  echo 'SUBSYSTEM==\"apex\", MODE=\"0660\", GROUP=\"plugdev\"' | "
                  "sudo tee /etc/udev/rules.d/99-apex.rules")
            print("  sudo udevadm control --reload-rules && sudo udevadm trigger")
            return False

    print("TPU access OK\n")
    return True


def find_models(model_dir: Path) -> list[Path]:
    """Find Edge TPU models in directory."""
    models = []
    for pattern in ["*_edgetpu.tflite", "*edgetpu*.tflite"]:
        models.extend(model_dir.glob(pattern))
    return sorted(set(models))


def main():
    parser = argparse.ArgumentParser(
        description="Benchmark Dual Coral Edge TPU",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )

    parser.add_argument(
        "--model", "-m",
        type=str,
        help="Path to Edge TPU model (.tflite). If not specified, benchmarks all models in models/"
    )
    parser.add_argument(
        "--model-dir",
        type=str,
        default="models",
        help="Directory containing models (default: models/)"
    )
    parser.add_argument(
        "--iterations", "-n",
        type=int,
        default=1000,
        help="Iterations per benchmark (default: 1000)"
    )
    parser.add_argument(
        "--sustained", "-s",
        type=int,
        default=60,
        help="Sustained load duration in seconds (default: 60)"
    )
    parser.add_argument(
        "--output-dir", "-o",
        type=str,
        default="benchmark_results",
        help="Output directory for results (default: benchmark_results/)"
    )
    parser.add_argument(
        "--quick",
        action="store_true",
        help="Quick benchmark (100 iterations, 10s sustained)"
    )
    parser.add_argument(
        "--stress",
        action="store_true",
        help="Extended stress test (5000 iterations, 5 min sustained)"
    )

    args = parser.parse_args()

    # Check TPU access
    if not check_prerequisites():
        return 1

    # Set parameters
    if args.quick:
        iterations = 100
        sustained = 10
    elif args.stress:
        iterations = 5000
        sustained = 300
    else:
        iterations = args.iterations
        sustained = args.sustained

    # Find models to benchmark
    model_dir = Path(args.model_dir)
    if args.model:
        models = [Path(args.model)]
    else:
        models = find_models(model_dir)
        if not models:
            print(f"No Edge TPU models found in {model_dir}/")
            print("\nDownload models:")
            print("  cd models/")
            print("  wget https://github.com/google-coral/test_data/raw/master/"
                  "ssd_mobilenet_v2_coco_quant_postprocess_edgetpu.tflite")
            return 1

    print(f"Models to benchmark: {len(models)}")
    for m in models:
        print(f"  {m.name}")
    print()

    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    all_results = []

    for model_path in models:
        if not model_path.exists():
            print(f"Model not found: {model_path}")
            continue

        print(f"\n{'#'*60}")
        print(f"# BENCHMARKING: {model_path.name}")
        print(f"{'#'*60}")

        try:
            benchmark = DualTPUBenchmark(str(model_path))
            results = benchmark.run_full_suite(
                iterations=iterations,
                sustained_duration=sustained
            )
            all_results.extend(results)

            # Save per-model results
            model_name = model_path.stem
            json_path = output_dir / f"{model_name}_{timestamp}.json"
            md_path = output_dir / f"{model_name}_{timestamp}.md"

            benchmark.save_results(str(json_path))
            benchmark.generate_markdown_report(str(md_path))

        except Exception as e:
            print(f"Benchmark failed for {model_path.name}: {e}")
            import traceback
            traceback.print_exc()

    # Summary
    print("\n" + "="*60)
    print("BENCHMARK COMPLETE")
    print("="*60)
    print(f"Results saved to: {output_dir}/")
    print(f"Total benchmarks run: {len(all_results)}")

    if all_results:
        print("\nHighlights:")
        # Find best throughput
        best = max(all_results, key=lambda r: r.throughput_fps)
        print(f"  Best throughput: {best.throughput_fps:.1f} inf/s ({best.name})")

        # Find lowest latency
        fastest = min(all_results, key=lambda r: r.latency_mean_ms)
        print(f"  Lowest latency: {fastest.latency_mean_ms:.2f}ms ({fastest.name})")

        # Max temp
        temps = [r.temp_max_c for r in all_results if r.temp_max_c]
        if temps:
            print(f"  Max temperature: {max(temps):.1f}C")

    return 0


if __name__ == "__main__":
    sys.exit(main())
