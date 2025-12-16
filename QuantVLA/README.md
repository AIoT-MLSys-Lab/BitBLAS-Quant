# QuantVLA Benchmark Reports

This folder contains sequential benchmark results for BitBLAS matmul on GR00T/PI layer configs, with both random and real activations/weights, and optional tuning.

## Environment
- CUDA 12.1 (PATH includes `/usr/local/cuda/bin`)
- TileLang/TVM built from `3rdparty/tilelang` and `3rdparty/tilelang/build/tvm` (set `PYTHONPATH`, `TILELANG_LIBRARY_PATH`, `TVM_LIBRARY_PATH`, `LD_LIBRARY_PATH` accordingly).
- Python: `.venv` in BitBLAS (torch 2.3.1+cu121).
- Cache dirs on scratch: `XDG_CACHE_HOME=/local/scratch1/wang.20306/.cache`, `HF_HOME` same, `PIP_CACHE_DIR` same, `TMPDIR=/local/scratch1/wang.20306/.tmp`.

## How to run (example)
```bash
cd /local/scratch1/wang.20306/BitBLAS
export CUDA_DEVICE_ORDER=PCI_BUS_ID
export PATH=/usr/local/cuda/bin:$PATH
export PYTHONPATH=$PWD:$PWD/3rdparty/tilelang:$PWD/3rdparty/tilelang/3rdparty/tvm/python:$PYTHONPATH
export TILELANG_LIBRARY_PATH=$PWD/3rdparty/tilelang/build
export TVM_LIBRARY_PATH=$PWD/3rdparty/tilelang/build/tvm
export LD_LIBRARY_PATH=$TILELANG_LIBRARY_PATH:$TVM_LIBRARY_PATH:$LD_LIBRARY_PATH

# W4A8 with real weights/activations, tuning enabled
./.venv/bin/python ../tools/pi_duquant_sequence_benchmark.py \
  --layer-log $PWD/QuantVLA/gr00tN1.5_llm_ditmlp_layer.json \
  --weight-index /local/scratch1/wang.20306/Isaac-GR00T/models/GR00T-N1.5-3B/models--nvidia--GR00T-N1.5-3B/snapshots/869830fc749c35f34771aa5209f923ac57e4564e/model.safetensors.index.json \
  --activation-cache /local/scratch1/wang.20306/Isaac-GR00T/activations/gr00t_gr1.pt \
  --warmup 1 --runs 3 --enable-tuning \
  --output reports/QuantVLA/gr00t_w4a8_seq_real_acts_tuned.md
```

## Files and notes
- `gr00t_w4a8_seq.md` / `gr00t_w4a4_seq.md` / `pi_w4a8_seq.md` / `pi_w4a4_seq.md`: Random activations/weights, no tuning (baseline).
- `gr00t_w4a8_seq_real.md` / `gr00t_w4a4_seq_real.md`: Real weights, random activations.
- `gr00t_w4a8_seq_real_acts.md` / `gr00t_w4a4_seq_real_acts.md`: Real weights + real activations (from demo data), no tuning.
- `gr00t_w4a8_seq_real_acts_tuned.md`: Real weights + real activations with tuning (Tensor Core search). Speedup ~0.82× vs FP16, accuracy improved vs untuned.

## Quick analysis
- Using real activations + row-wise weight scales reduces error significantly versus random inputs.
- Tuning (`--enable-tuning`) improves accuracy and slightly improves speed, but overall speedup is still <1× FP16 for small batch/sequence shapes (launch/scale overhead dominates).
- To further reduce error/boost speed: use more calibration samples, try group-size (e.g., 128) quantization, and batch/layer fusion to amortize overhead.
