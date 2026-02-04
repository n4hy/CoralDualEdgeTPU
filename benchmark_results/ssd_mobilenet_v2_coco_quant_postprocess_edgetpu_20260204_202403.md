# Dual Coral Edge TPU Benchmark Results

**Generated:** 2026-02-04 20:26:08

## System Configuration

- **Platform:** orangepi6plus (aarch64)
- **Kernel:** 6.6.89-cix
- **TPU Devices:** 2
  - TPU 0: `/dev/apex_0`
  - TPU 1: `/dev/apex_1`

## Model Information

- **Model:** `ssd_mobilenet_v2_coco_quant_postprocess_edgetpu`
- **Input Shape:** `(1, 300, 300, 3)`
- **Input Type:** `<class 'numpy.uint8'>`

## Results Summary

| Test | Throughput (inf/s) | Latency Mean (ms) | Latency p99 (ms) | Temp Max (C) |
|------|-------------------|-------------------|------------------|--------------|
| single_tpu_0 | 102.2 | 9.71 | 11.44 | 42.0 |
| single_tpu_1 | 112.7 | 8.81 | 11.37 | 42.0 |
| dual_tpu_parallel | 213.7 | 9.15 | 11.96 | 42.0 |
| dual_tpu_alternating | 100.3 | 9.91 | 11.63 | 42.0 |
| sustained_load_60s | 210.4 | 9.50 | 12.96 | 42.0 |

## Detailed Results

### single_tpu_0

- **Iterations:** 1000
- **Total Time:** 9.79s
- **Throughput:** 102.2 inferences/second

**Latency Distribution (ms):**

| Metric | Value |
|--------|-------|
| Mean | 9.71 |
| Std Dev | 1.39 |
| Min | 6.54 |
| Max | 12.73 |
| p50 | 9.27 |
| p95 | 11.37 |
| p99 | 11.44 |

**Thermal:**

- Start: 42.0C
- End: 42.0C
- Max: 42.0C

### single_tpu_1

- **Iterations:** 1000
- **Total Time:** 8.88s
- **Throughput:** 112.7 inferences/second

**Latency Distribution (ms):**

| Metric | Value |
|--------|-------|
| Mean | 8.81 |
| Std Dev | 1.27 |
| Min | 6.55 |
| Max | 11.44 |
| p50 | 8.84 |
| p95 | 11.21 |
| p99 | 11.37 |

**Thermal:**

- Start: 42.0C
- End: 42.0C
- Max: 42.0C

### dual_tpu_parallel

- **Iterations:** 2000
- **Total Time:** 9.36s
- **Throughput:** 213.7 inferences/second

**Latency Distribution (ms):**

| Metric | Value |
|--------|-------|
| Mean | 9.15 |
| Std Dev | 1.05 |
| Min | 6.52 |
| Max | 14.23 |
| p50 | 9.02 |
| p95 | 11.33 |
| p99 | 11.96 |

**Thermal:**

- Start: 42.0C
- End: 42.0C
- Max: 42.0C

### dual_tpu_alternating

- **Iterations:** 1000
- **Total Time:** 9.97s
- **Throughput:** 100.3 inferences/second

**Latency Distribution (ms):**

| Metric | Value |
|--------|-------|
| Mean | 9.91 |
| Std Dev | 1.65 |
| Min | 6.54 |
| Max | 13.19 |
| p50 | 11.07 |
| p95 | 11.52 |
| p99 | 11.63 |

**Thermal:**

- Start: 42.0C
- End: 42.0C
- Max: 42.0C

### sustained_load_60s

- **Iterations:** 12846
- **Total Time:** 61.04s
- **Throughput:** 210.4 inferences/second

**Latency Distribution (ms):**

| Metric | Value |
|--------|-------|
| Mean | 9.50 |
| Std Dev | 1.43 |
| Min | 6.54 |
| Max | 16.98 |
| p50 | 9.23 |
| p95 | 11.76 |
| p99 | 12.96 |

**Thermal:**

- Start: 42.0C
- End: 42.0C
- Max: 42.0C
