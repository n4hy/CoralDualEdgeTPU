# Dual Coral Edge TPU Benchmark Results

**Generated:** 2026-02-01 02:01:53

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
| single_tpu_0 | 112.4 | 8.83 | 11.53 | 43.0 |
| single_tpu_1 | 112.7 | 8.80 | 11.54 | 43.0 |
| dual_tpu_parallel | 216.2 | 9.20 | 12.68 | 43.0 |
| dual_tpu_alternating | 110.9 | 8.95 | 11.53 | 43.0 |
| sustained_load_300s | 205.3 | 9.74 | 15.06 | 43.0 |

## Detailed Results

### single_tpu_0

- **Iterations:** 5000
- **Total Time:** 44.50s
- **Throughput:** 112.4 inferences/second

**Latency Distribution (ms):**

| Metric | Value |
|--------|-------|
| Mean | 8.83 |
| Std Dev | 1.00 |
| Min | 6.53 |
| Max | 16.21 |
| p50 | 8.84 |
| p95 | 11.11 |
| p99 | 11.53 |

**Thermal:**

- Start: 43.0C
- End: 43.0C
- Max: 43.0C

### single_tpu_1

- **Iterations:** 5000
- **Total Time:** 44.37s
- **Throughput:** 112.7 inferences/second

**Latency Distribution (ms):**

| Metric | Value |
|--------|-------|
| Mean | 8.80 |
| Std Dev | 1.04 |
| Min | 6.48 |
| Max | 15.26 |
| p50 | 8.81 |
| p95 | 11.29 |
| p99 | 11.54 |

**Thermal:**

- Start: 43.0C
- End: 43.0C
- Max: 43.0C

### dual_tpu_parallel

- **Iterations:** 10000
- **Total Time:** 46.24s
- **Throughput:** 216.2 inferences/second

**Latency Distribution (ms):**

| Metric | Value |
|--------|-------|
| Mean | 9.20 |
| Std Dev | 1.29 |
| Min | 6.45 |
| Max | 14.93 |
| p50 | 9.06 |
| p95 | 11.56 |
| p99 | 12.68 |

**Thermal:**

- Start: 43.0C
- End: 43.0C
- Max: 43.0C

### dual_tpu_alternating

- **Iterations:** 5000
- **Total Time:** 45.08s
- **Throughput:** 110.9 inferences/second

**Latency Distribution (ms):**

| Metric | Value |
|--------|-------|
| Mean | 8.95 |
| Std Dev | 1.17 |
| Min | 6.50 |
| Max | 13.83 |
| p50 | 8.92 |
| p95 | 11.31 |
| p99 | 11.53 |

**Thermal:**

- Start: 43.0C
- End: 43.0C
- Max: 43.0C

### sustained_load_300s

- **Iterations:** 62514
- **Total Time:** 304.55s
- **Throughput:** 205.3 inferences/second

**Latency Distribution (ms):**

| Metric | Value |
|--------|-------|
| Mean | 9.74 |
| Std Dev | 1.85 |
| Min | 6.54 |
| Max | 22.58 |
| p50 | 9.24 |
| p95 | 13.02 |
| p99 | 15.06 |

**Thermal:**

- Start: 43.0C
- End: 43.0C
- Max: 43.0C
