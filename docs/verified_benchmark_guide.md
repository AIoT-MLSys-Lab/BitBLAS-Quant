# Verified Dequant Benchmark Guide

This guide describes the updated benchmark scripts which now include automatic output verification to ensure data correctness.

## Overview

The benchmark scripts (`tools/dequant_benchmark.py` and `tools/dequant_sequence.py`) have been modified to perform a validation step before measuring performance. This prevents the recording of invalid results (e.g., "fake" speedups caused by kernel failures).

## How it Works

1.  **Validation Run**: Before the timing loop, the script runs the quantized matrix multiplication once.
2.  **Zero Check**: It checks if the output tensor contains all zeros.
3.  **Abort**: If the output is all zeros, the script raises a `RuntimeError` and aborts immediately.

## Usage

Run the benchmarks as usual. No new arguments are required.

### Shape-collapsed Benchmark

```bash
export PATH=/usr/local/cuda/bin:$PATH
export TVM_TARGET="cuda -arch=sm_86"  # Adjust based on your GPU

python tools/dequant_benchmark.py \
  --layer-log pi_llm_ditmlp_layers.json \
  --batch 1 \
  --warmup 5 \
  --runs 20 \
  --output docs/pi_duquant_w4a8.md
```

### Sequential Benchmark

```bash
python tools/dequant_sequence.py \
  --layer-log pi_llm_ditmlp_layers.json \
  --batch 1 \
  --warmup 2 \
  --runs 8 \
  --output docs/pi_duquant_w4a8_sequence.md
```

## Error Handling

If you see the following error:

```text
RuntimeError: Benchmark validation failed: Output is all zeros for shape M=..., N=..., K=...
This indicates the kernel is not executing correctly (likely architecture mismatch).
```

**Meaning**: The GPU kernel failed to compute valid results. This usually happens when the compiled CUDA kernel target (e.g., `sm_86`) is not compatible with the actual GPU architecture (e.g., `sm_110` Blackwell), or if there is a toolchain mismatch.

**Action**:
- Check your `TVM_TARGET` environment variable.
- Ensure your CUDA toolkit version supports your GPU.
- Do **not** use the results if this error occurs.
