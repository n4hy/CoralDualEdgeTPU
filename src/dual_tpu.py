"""Dual Coral Edge TPU management interface.

Provides load-balanced inference across two PCIe Coral Edge TPUs.
Uses tflite-runtime with Edge TPU delegate directly (no pycoral dependency).
"""

import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np
from tflite_runtime.interpreter import Interpreter, load_delegate


def list_edge_tpus() -> list[str]:
    """List available Edge TPU device paths."""
    devices = []
    for i in range(8):  # Check up to 8 devices
        path = f"/dev/apex_{i}"
        if Path(path).exists():
            devices.append(path)
    return devices


def make_interpreter(model_path: str, device: Optional[str] = None) -> Interpreter:
    """Create an interpreter with Edge TPU delegate.

    Args:
        model_path: Path to Edge TPU compiled .tflite model
        device: Device path (e.g., "/dev/apex_0") or None for default

    Returns:
        TFLite Interpreter with Edge TPU delegate
    """
    delegate_options = {}
    if device:
        # Extract device index from path like "/dev/apex_0" -> "0"
        if device.startswith("/dev/apex_"):
            device_idx = device.replace("/dev/apex_", "")
            delegate_options["device"] = f":{device_idx}"
        else:
            delegate_options["device"] = device

    delegates = [load_delegate("libedgetpu.so.1", delegate_options)]
    return Interpreter(model_path=model_path, experimental_delegates=delegates)


@dataclass
class TPUDevice:
    """Represents a single Edge TPU device."""
    device_path: str
    interpreter: Optional[Interpreter] = None
    model_path: Optional[str] = None
    lock: threading.Lock = None

    def __post_init__(self):
        self.lock = threading.Lock()


class DualEdgeTPU:
    """Manages dual Coral Edge TPU devices for parallel inference.

    Supports:
    - Independent models on each TPU
    - Load-balanced inference with same model
    - Parallel inference for different tasks (e.g., detection + classification)
    """

    def __init__(self):
        self.devices: list[TPUDevice] = []
        self._round_robin_idx = 0
        self._rr_lock = threading.Lock()
        self._discover_devices()

    def _discover_devices(self) -> None:
        """Discover available Edge TPU devices."""
        tpus = list_edge_tpus()
        if not tpus:
            raise RuntimeError(
                "No Edge TPU devices found. Check:\n"
                "  1. Devices exist: ls /dev/apex*\n"
                "  2. Permissions: user in 'plugdev' group\n"
                "  3. Driver loaded: lsmod | grep apex"
            )

        for device_path in tpus:
            self.devices.append(TPUDevice(device_path=device_path))

        print(f"Discovered {len(self.devices)} Edge TPU(s): "
              f"{[d.device_path for d in self.devices]}")

    @property
    def num_devices(self) -> int:
        """Number of available TPU devices."""
        return len(self.devices)

    def load_model(self, model_path: str, device_idx: Optional[int] = None) -> None:
        """Load a compiled Edge TPU model.

        Args:
            model_path: Path to .tflite model (must be Edge TPU compiled)
            device_idx: Specific device index, or None to load on all devices
        """
        model_path = str(Path(model_path).resolve())

        if device_idx is not None:
            self._load_model_on_device(model_path, device_idx)
        else:
            for idx in range(len(self.devices)):
                self._load_model_on_device(model_path, idx)

    def _load_model_on_device(self, model_path: str, device_idx: int) -> None:
        """Load model on a specific device."""
        device = self.devices[device_idx]
        with device.lock:
            interpreter = make_interpreter(model_path, device=device.device_path)
            interpreter.allocate_tensors()
            device.interpreter = interpreter
            device.model_path = model_path
            print(f"Loaded {Path(model_path).name} on {device.device_path}")

    def infer(self, input_data: np.ndarray, device_idx: Optional[int] = None) -> np.ndarray:
        """Run inference on specified device or round-robin if not specified.

        Args:
            input_data: Preprocessed input tensor matching model input shape
            device_idx: Specific device, or None for automatic load balancing

        Returns:
            Model output tensor
        """
        if device_idx is None:
            device_idx = self._get_next_device()

        device = self.devices[device_idx]
        with device.lock:
            if device.interpreter is None:
                raise RuntimeError(f"No model loaded on device {device_idx}")

            # Set input
            input_details = device.interpreter.get_input_details()[0]
            device.interpreter.set_tensor(input_details["index"],
                                          np.expand_dims(input_data, axis=0))

            # Run inference
            device.interpreter.invoke()

            # Get output
            output_details = device.interpreter.get_output_details()[0]
            return device.interpreter.get_tensor(output_details["index"]).copy()

    def _get_next_device(self) -> int:
        """Get next device index using round-robin."""
        with self._rr_lock:
            idx = self._round_robin_idx
            self._round_robin_idx = (self._round_robin_idx + 1) % len(self.devices)
            return idx

    def classify(self, input_data: np.ndarray, device_idx: Optional[int] = None,
                 top_k: int = 5, threshold: float = 0.0) -> list:
        """Run classification inference.

        Args:
            input_data: Preprocessed input image
            device_idx: Specific device or None for load balancing
            top_k: Number of top results to return
            threshold: Minimum score threshold

        Returns:
            List of (class_id, score) tuples
        """
        if device_idx is None:
            device_idx = self._get_next_device()

        device = self.devices[device_idx]
        with device.lock:
            if device.interpreter is None:
                raise RuntimeError(f"No model loaded on device {device_idx}")

            # Set input
            input_details = device.interpreter.get_input_details()[0]
            device.interpreter.set_tensor(input_details["index"],
                                          np.expand_dims(input_data, axis=0))

            # Run inference
            device.interpreter.invoke()

            # Get output scores
            output_details = device.interpreter.get_output_details()[0]
            scores = device.interpreter.get_tensor(output_details["index"])[0]

            # Dequantize if needed
            if output_details["dtype"] == np.uint8:
                scale, zero_point = output_details["quantization"]
                scores = (scores.astype(np.float32) - zero_point) * scale

            # Get top-k results
            top_indices = np.argsort(scores)[::-1][:top_k]
            results = []
            for idx in top_indices:
                score = float(scores[idx])
                if score >= threshold:
                    results.append(ClassResult(id=int(idx), score=score))

            return results

    def detect(self, input_data: np.ndarray, device_idx: Optional[int] = None,
               threshold: float = 0.5) -> list:
        """Run object detection inference.

        Args:
            input_data: Preprocessed input image
            device_idx: Specific device or None for load balancing
            threshold: Minimum detection score

        Returns:
            List of detected objects with bounding boxes
        """
        if device_idx is None:
            device_idx = self._get_next_device()

        device = self.devices[device_idx]
        with device.lock:
            if device.interpreter is None:
                raise RuntimeError(f"No model loaded on device {device_idx}")

            # Set input
            input_details = device.interpreter.get_input_details()[0]
            device.interpreter.set_tensor(input_details["index"],
                                          np.expand_dims(input_data, axis=0))

            # Run inference
            device.interpreter.invoke()

            # Get outputs - SSD MobileNet format
            output_details = device.interpreter.get_output_details()

            # Standard SSD output format:
            # 0: bounding boxes [1, num_detections, 4] (ymin, xmin, ymax, xmax normalized)
            # 1: class IDs [1, num_detections]
            # 2: scores [1, num_detections]
            # 3: num detections [1]

            boxes = device.interpreter.get_tensor(output_details[0]["index"])[0]
            class_ids = device.interpreter.get_tensor(output_details[1]["index"])[0]
            scores = device.interpreter.get_tensor(output_details[2]["index"])[0]
            num_detections = int(device.interpreter.get_tensor(output_details[3]["index"])[0])

            results = []
            for i in range(num_detections):
                score = float(scores[i])
                if score >= threshold:
                    ymin, xmin, ymax, xmax = boxes[i]
                    results.append(DetectionResult(
                        id=int(class_ids[i]),
                        score=score,
                        bbox=BBox(xmin=float(xmin), ymin=float(ymin),
                                  xmax=float(xmax), ymax=float(ymax))
                    ))

            return results

    def parallel_infer(self, inputs: list[tuple[np.ndarray, int]]) -> list[np.ndarray]:
        """Run inference on multiple inputs in parallel across devices.

        Args:
            inputs: List of (input_data, device_idx) tuples

        Returns:
            List of outputs in same order as inputs
        """
        results = [None] * len(inputs)
        threads = []

        def run_inference(idx, data, dev_idx):
            results[idx] = self.infer(data, dev_idx)

        for i, (data, dev_idx) in enumerate(inputs):
            t = threading.Thread(target=run_inference, args=(i, data, dev_idx))
            threads.append(t)
            t.start()

        for t in threads:
            t.join()

        return results

    def get_input_details(self, device_idx: int = 0) -> dict:
        """Get input tensor details for loaded model."""
        device = self.devices[device_idx]
        if device.interpreter is None:
            raise RuntimeError(f"No model loaded on device {device_idx}")
        return device.interpreter.get_input_details()[0]

    def get_output_details(self, device_idx: int = 0) -> dict:
        """Get output tensor details for loaded model."""
        device = self.devices[device_idx]
        if device.interpreter is None:
            raise RuntimeError(f"No model loaded on device {device_idx}")
        return device.interpreter.get_output_details()[0]


@dataclass
class BBox:
    """Bounding box in normalized coordinates."""
    xmin: float
    ymin: float
    xmax: float
    ymax: float


@dataclass
class DetectionResult:
    """Object detection result."""
    id: int
    score: float
    bbox: BBox


@dataclass
class ClassResult:
    """Classification result."""
    id: int
    score: float


def check_tpu_status() -> dict:
    """Check Edge TPU device status.

    Returns:
        Dict with device info and status
    """
    import subprocess
    import os

    status = {
        "devices": [],
        "driver_loaded": False,
        "pcie_devices": []
    }

    # Check device nodes
    for apex in Path("/dev").glob("apex_*"):
        status["devices"].append({
            "path": str(apex),
            "readable": os.access(apex, os.R_OK),
            "writable": os.access(apex, os.W_OK),
        })

    # Check driver
    try:
        result = subprocess.run(
            ["lsmod"], capture_output=True, text=True, check=True
        )
        status["driver_loaded"] = "apex" in result.stdout
    except subprocess.CalledProcessError:
        pass

    # Check PCIe
    try:
        result = subprocess.run(
            ["lspci", "-d", "1ac1:089a"],
            capture_output=True, text=True, check=True
        )
        status["pcie_devices"] = [
            line.strip() for line in result.stdout.strip().split("\n") if line
        ]
    except subprocess.CalledProcessError:
        pass

    return status
