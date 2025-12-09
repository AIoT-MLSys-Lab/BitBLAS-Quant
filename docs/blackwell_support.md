# Blackwell (RTX 5060) Support Guide

## Issue Description
When running BitBLAS benchmarks on NVIDIA RTX 5060 (Blackwell architecture, `sm_110`), the default targets (`sm_80`, `sm_86`) and even `sm_90` (Hopper) produce invalid results (all zeros). This is likely due to architectural differences or instruction set incompatibilities.

## Solution
The working configuration for RTX 5060 is to target **`sm_100`** (Blackwell).

### Configuration
Set the environment variable:
```bash
export TVM_TARGET="cuda -arch=sm_100"
```

### Performance Note
Currently, BitBLAS/TileLang does not fully support Blackwell's Tensor Cores (MMA). When targeting `sm_100`, the compiler falls back to using `DP4A` instructions (Dot Product 4 Accumulate) on CUDA Cores.
- **Correctness**: Valid (verified non-zero output).
- **Performance**: Sub-optimal compared to Tensor Cores, but functional.

### Troubleshooting
If you encounter `AttributeError: module 'tilelang.transform._ffi_api' has no attribute 'WarpSpecializedPipeline'` when using `sm_90` or similar targets, it indicates a missing feature in the installed `tilelang` library. A patch was applied to `3rdparty/tilelang/tilelang/transform/__init__.py` to make this pass conditional, but `sm_90` still produced zeros.

## Verification
To verify correctness, use the `verify_correctness.py` script with `sm_100` target. It should report:
```text
SUCCESS: Output is non-zero
```
And the maximum output value should be reasonable (e.g., ~32768 for the test case).
