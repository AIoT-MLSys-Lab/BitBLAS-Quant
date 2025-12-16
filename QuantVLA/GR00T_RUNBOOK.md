# GR00T: How to capture activations and run BitBLAS benchmarks

## 1) Capture GR00T activations
Prereqs: Isaac-GR00T repo and GR00T-N1.5-3B weights already downloaded (snapshot path below).

```bash
cd /local/scratch1/wang.20306/Isaac-GR00T
env TMPDIR=/local/scratch1/wang.20306/.tmp \
    XDG_CACHE_HOME=/local/scratch1/wang.20306/.cache \
    HF_HOME=/local/scratch1/wang.20306/.cache/huggingface \
    PIP_CACHE_DIR=/local/scratch1/wang.20306/.cache/pip \
    ../BitBLAS/.venv/bin/python scripts/capture_activations.py \
      --model-path models/GR00T-N1.5-3B/models--nvidia--GR00T-N1.5-3B/snapshots/869830fc749c35f34771aa5209f923ac57e4564e \
      --dataset-path demo_data/robot_sim.PickNPlace \
      --num-samples 1 \
      --video-backend decord \
      --output activations/gr00t_gr1.pt
```
Output: `Isaac-GR00T/activations/gr00t_gr1.pt` (contains per-layer Linear inputs and stats).

## 2) Run BitBLAS sequential benchmarks with real weights/activations
Set env (TileLang/TVM + CUDA):
```bash
cd /local/scratch1/wang.20306/BitBLAS
export CUDA_DEVICE_ORDER=PCI_BUS_ID
export PATH=/usr/local/cuda/bin:$PATH
export PYTHONPATH=$PWD:$PWD/3rdparty/tilelang:$PWD/3rdparty/tilelang/3rdparty/tvm/python:$PYTHONPATH
export TILELANG_LIBRARY_PATH=$PWD/3rdparty/tilelang/build
export TVM_LIBRARY_PATH=$PWD/3rdparty/tilelang/build/tvm
export LD_LIBRARY_PATH=$TILELANG_LIBRARY_PATH:$TVM_LIBRARY_PATH:$LD_LIBRARY_PATH
```
Run W4A8 with tuning (Tensor Core search):
```bash
./.venv/bin/python ../tools/pi_duquant_sequence_benchmark.py \
  --layer-log $PWD/QuantVLA/gr00tN1.5_llm_ditmlp_layer.json \
  --weight-index /local/scratch1/wang.20306/Isaac-GR00T/models/GR00T-N1.5-3B/models--nvidia--GR00T-N1.5-3B/snapshots/869830fc749c35f34771aa5209f923ac57e4564e/model.safetensors.index.json \
  --activation-cache /local/scratch1/wang.20306/Isaac-GR00T/activations/gr00t_gr1.pt \
  --warmup 1 --runs 3 --enable-tuning \
  --output reports/QuantVLA/gr00t_w4a8_seq_real_acts_tuned.md
```
Run W4A4 (4/4 bit):
```bash
./.venv/bin/python ../tools/pi_duquant_sequence_benchmark.py \
  --layer-log $PWD/QuantVLA/gr00tN1.5_llm_ditmlp_layer.json \
  --weight-index /local/scratch1/wang.20306/Isaac-GR00T/models/GR00T-N1.5-3B/models--nvidia--GR00T-N1.5-3B/snapshots/869830fc749c35f34771aa5209f923ac57e4564e/model.safetensors.index.json \
  --activation-cache /local/scratch1/wang.20306/Isaac-GR00T/activations/gr00t_gr1.pt \
  --act-max-q 7 --weight-max-q 7 \
  --warmup 1 --runs 3 --enable-tuning \
  --output reports/QuantVLA/gr00t_w4a4_seq_real_acts_tuned.md
```

## 3) Where to find results
- Reports in this folder: random baselines, real-weight baselines, real-activation runs, and tuned runs.
- Key tuned file: `gr00t_w4a8_seq_real_acts_tuned.md` (current best accuracy; speed ~0.82× FP16).
- Activation cache: `/local/scratch1/wang.20306/Isaac-GR00T/activations/gr00t_gr1.pt`
- Model weights index: `/local/scratch1/wang.20306/Isaac-GR00T/models/GR00T-N1.5-3B/.../model.safetensors.index.json`

## Notes
- First tuned run compiles/searches kernels and can be slow; reruns reuse cached kernels.
- Small batch/seq shapes still see <1× speedup due to launch/scale overhead; more calibration samples or group-size quantization may reduce error further.
