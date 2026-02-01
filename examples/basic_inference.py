#!/usr/bin/env python3
"""Basic example of dual Edge TPU inference.

Demonstrates:
- Device discovery
- Model loading on both TPUs
- Load-balanced classification
- Parallel detection + classification
"""

import sys
from pathlib import Path

import numpy as np
from PIL import Image

# Add src to path for local development
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.dual_tpu import DualEdgeTPU, check_tpu_status


def preprocess_image(image_path: str, target_size: tuple[int, int]) -> np.ndarray:
    """Load and preprocess an image for inference."""
    img = Image.open(image_path).convert("RGB")
    img = img.resize(target_size, Image.LANCZOS)
    return np.array(img, dtype=np.uint8)


def main():
    # Check TPU status first
    print("Checking TPU status...")
    status = check_tpu_status()
    print(f"  Devices found: {len(status['devices'])}")
    print(f"  Driver loaded: {status['driver_loaded']}")
    print(f"  PCIe devices: {status['pcie_devices']}")

    if not status["devices"]:
        print("\nNo Edge TPU devices found!")
        print("Run: ls -la /dev/apex*")
        return 1

    # Initialize dual TPU manager
    print("\nInitializing DualEdgeTPU...")
    try:
        tpu = DualEdgeTPU()
    except RuntimeError as e:
        print(f"Failed to initialize: {e}")
        return 1

    print(f"Found {tpu.num_devices} TPU device(s)")

    # Example: Load classification model on both TPUs
    # Download models from: https://coral.ai/models/
    model_path = Path(__file__).parent.parent / "models" / "mobilenet_v2_1.0_224_inat_bird_quant_edgetpu.tflite"

    if not model_path.exists():
        print(f"\nModel not found: {model_path}")
        print("Download a model from https://coral.ai/models/")
        print("Example:")
        print("  cd models/")
        print("  wget https://github.com/google-coral/test_data/raw/master/mobilenet_v2_1.0_224_inat_bird_quant_edgetpu.tflite")
        return 1

    print(f"\nLoading model on all TPUs...")
    tpu.load_model(str(model_path))

    # Get input shape
    input_details = tpu.get_input_details()
    input_shape = input_details["shape"]
    print(f"Model input shape: {input_shape}")

    # Example inference with dummy data
    dummy_input = np.random.randint(0, 255, size=input_shape[1:], dtype=np.uint8)

    print("\nRunning inference on TPU 0...")
    result0 = tpu.classify(dummy_input, device_idx=0, top_k=3)
    print(f"  Top-3 classes: {result0}")

    print("\nRunning inference on TPU 1...")
    result1 = tpu.classify(dummy_input, device_idx=1, top_k=3)
    print(f"  Top-3 classes: {result1}")

    # Load-balanced inference
    print("\nRunning 10 load-balanced inferences...")
    for i in range(10):
        result = tpu.classify(dummy_input, top_k=1)
        print(f"  Inference {i}: class {result[0].id if result else 'none'}")

    print("\nDual TPU test complete!")
    return 0


if __name__ == "__main__":
    sys.exit(main())
