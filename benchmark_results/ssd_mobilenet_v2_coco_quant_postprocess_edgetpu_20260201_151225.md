# Dual Coral Edge TPU Benchmark Results

**Generated:** 2026-02-01 15:14:27

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
| single_tpu_0 | 110.6 | 8.97 | 12.12 | 42.0 |
| single_tpu_1 | 110.0 | 9.03 | 12.11 | 41.0 |
| dual_tpu_parallel | 217.1 | 9.16 | 12.81 | 41.0 |
| dual_tpu_alternating | 110.8 | 8.96 | 12.50 | 41.0 |
| sustained_load_60s | 219.1 | 9.12 | 12.89 | 41.0 |

## Detailed Results

### single_tpu_0

- **Iterations:** 1000
- **Total Time:** 9.04s
- **Throughput:** 110.6 inferences/second

**Latency Distribution (ms):**

| Metric | Value |
|--------|-------|
| Mean | 8.97 |
| Std Dev | 0.93 |
| Min | 6.52 |
| Max | 15.53 |
| p50 | 8.87 |
| p95 | 11.02 |
| p99 | 12.12 |

**Thermal:**

- Start: 42.0C
- End: 41.0C
- Max: 42.0C

### single_tpu_1

- **Iterations:** 1000
- **Total Time:** 9.09s
- **Throughput:** 110.0 inferences/second

**Latency Distribution (ms):**

| Metric | Value |
|--------|-------|
| Mean | 9.03 |
| Std Dev | 0.97 |
| Min | 6.54 |
| Max | 17.82 |
| p50 | 8.99 |
| p95 | 11.08 |
| p99 | 12.11 |

**Thermal:**

- Start: 41.0C
- End: 41.0C
- Max: 41.0C

### dual_tpu_parallel

- **Iterations:** 2000
- **Total Time:** 9.21s
- **Throughput:** 217.1 inferences/second

**Latency Distribution (ms):**

| Metric | Value |
|--------|-------|
| Mean | 9.16 |
| Std Dev | 1.06 |
| Min | 6.63 |
| Max | 15.93 |
| p50 | 9.07 |
| p95 | 11.18 |
| p99 | 12.81 |

**Thermal:**

- Start: 41.0C
- End: 41.0C
- Max: 41.0C

### dual_tpu_alternating

- **Iterations:** 1000
- **Total Time:** 9.03s
- **Throughput:** 110.8 inferences/second

**Latency Distribution (ms):**

| Metric | Value |
|--------|-------|
| Mean | 8.96 |
| Std Dev | 0.94 |
| Min | 6.62 |
| Max | 15.91 |
| p50 | 8.94 |
| p95 | 10.97 |
| p99 | 12.50 |

**Thermal:**

- Start: 41.0C
- End: 41.0C
- Max: 41.0C

### sustained_load_60s

- **Iterations:** 13358
- **Total Time:** 60.97s
- **Throughput:** 219.1 inferences/second

**Latency Distribution (ms):**

| Metric | Value |
|--------|-------|
| Mean | 9.12 |
| Std Dev | 1.12 |
| Min | 6.49 |
| Max | 24.55 |
| p50 | 9.04 |
| p95 | 11.22 |
| p99 | 12.89 |

**Thermal:**

- Start: 41.0C
- End: 41.0C
- Max: 41.0C
