# Dual Coral Edge TPU Benchmark Results

**Generated:** 2026-02-01 01:53:16

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
| single_tpu_0 | 114.7 | 8.66 | 11.42 | 43.0 |
| single_tpu_1 | 109.6 | 9.07 | 11.53 | 43.0 |
| dual_tpu_parallel | 217.4 | 9.14 | 12.84 | 43.0 |
| dual_tpu_alternating | 109.2 | 9.10 | 11.63 | 43.0 |
| sustained_load_60s | 219.2 | 9.12 | 12.52 | 43.0 |

## Detailed Results

### single_tpu_0

- **Iterations:** 1000
- **Total Time:** 8.72s
- **Throughput:** 114.7 inferences/second

**Latency Distribution (ms):**

| Metric | Value |
|--------|-------|
| Mean | 8.66 |
| Std Dev | 0.92 |
| Min | 6.53 |
| Max | 14.32 |
| p50 | 8.79 |
| p95 | 9.78 |
| p99 | 11.42 |

**Thermal:**

- Start: 43.0C
- End: 43.0C
- Max: 43.0C

### single_tpu_1

- **Iterations:** 1000
- **Total Time:** 9.12s
- **Throughput:** 109.6 inferences/second

**Latency Distribution (ms):**

| Metric | Value |
|--------|-------|
| Mean | 9.07 |
| Std Dev | 1.14 |
| Min | 6.60 |
| Max | 15.69 |
| p50 | 8.93 |
| p95 | 11.36 |
| p99 | 11.53 |

**Thermal:**

- Start: 43.0C
- End: 43.0C
- Max: 43.0C

### dual_tpu_parallel

- **Iterations:** 2000
- **Total Time:** 9.20s
- **Throughput:** 217.4 inferences/second

**Latency Distribution (ms):**

| Metric | Value |
|--------|-------|
| Mean | 9.14 |
| Std Dev | 1.20 |
| Min | 6.58 |
| Max | 15.49 |
| p50 | 9.04 |
| p95 | 11.51 |
| p99 | 12.84 |

**Thermal:**

- Start: 43.0C
- End: 43.0C
- Max: 43.0C

### dual_tpu_alternating

- **Iterations:** 1000
- **Total Time:** 9.16s
- **Throughput:** 109.2 inferences/second

**Latency Distribution (ms):**

| Metric | Value |
|--------|-------|
| Mean | 9.10 |
| Std Dev | 1.14 |
| Min | 6.57 |
| Max | 13.22 |
| p50 | 9.05 |
| p95 | 11.35 |
| p99 | 11.63 |

**Thermal:**

- Start: 43.0C
- End: 43.0C
- Max: 43.0C

### sustained_load_60s

- **Iterations:** 13355
- **Total Time:** 60.92s
- **Throughput:** 219.2 inferences/second

**Latency Distribution (ms):**

| Metric | Value |
|--------|-------|
| Mean | 9.12 |
| Std Dev | 1.16 |
| Min | 6.53 |
| Max | 17.63 |
| p50 | 9.03 |
| p95 | 11.41 |
| p99 | 12.52 |

**Thermal:**

- Start: 43.0C
- End: 43.0C
- Max: 43.0C
