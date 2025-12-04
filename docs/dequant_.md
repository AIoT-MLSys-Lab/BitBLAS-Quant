## DuQuant Benchmark Playbook

This repo now has two helper scripts under `tools/` for profiling mock W/A workloads.
They auto-detect the weight/activation precision in each DUQUANT log line
(`... W4 A8 ...`, `... W4 A4 ...`, etc.), so producing a new variant is as easy as
copying the log (e.g., `pi_llm_ditmlp_layers_w4a4.json`) and editing the `A8` token.
Both expect the `.venv` you prepared plus BitBLAS’ CUDA/TVM/TileLang environment.

### Pre-flight

```bash
source .venv/bin/activate
export PATH=/usr/local/cuda/bin:$PATH
export TVM_HOME=$(pwd)/3rdparty/tvm
export TILELANG_HOME=$(pwd)/3rdparty/tilelang
export PYTHONPATH=$TVM_HOME/python:$TILELANG_HOME:$(pwd):$PYTHONPATH
export CUDA_DEVICE_ORDER=PCI_BUS_ID
```

### 1. Shape-collapsed benchmark (`tools/dequant_benchmark.py`)

Use this when you only need a per-shape average plus a count-weighted speedup.
It deduplicates layer shapes extracted from a DUQUANT log.

```bash
python tools/dequant_benchmark.py \
  --layer-log pi_llm_ditmlp_layers.json \
  --batch 1 \
  --warmup 5 \
  --runs 20 \
  --output docs/pi_duquant_w4a8.md

python tools/dequant_benchmark.py \
  --layer-log gr00tN1.5_llm_ditmlp_layer.json \
  --batch 1 \
  --warmup 5 \
  --runs 20 \
  --output docs/gr00t_duquant_w4a8.md
```

Swap `--layer-log`/`--output` for other models, e.g. `gr00tN1.5_llm_ditmlp_layer.json`
or the `_w4a4` variants.
Key arguments:

- `--batch`: M dimension (default 1). Increase if you want batched matmuls.
- `--warmup` / `--runs`: CUDA-event warmup and measured iterations per shape.
- `--seed`: optional reproducibility guard for mock tensors.

### 2. Sequential benchmark (`tools/dequant_sequence.py`)

Runs every layer listed in the log **in order**, capturing per-layer timing and totals.
This highlights whether a specific bottleneck layer benefits from the chosen quant mode.

```bash
python tools/dequant_sequence.py \
  --layer-log pi_llm_ditmlp_layers.json \
  --batch 1 \
  --warmup 2 \
  --runs 8 \
  --output docs/pi_duquant_w4a8_sequence.md

python tools/dequant_sequence.py \
  --layer-log gr00tN1.5_llm_ditmlp_layer.json \
  --batch 1 \
  --warmup 2 \
  --runs 8 \
  --output docs/gr00t_duquant_w4a8_sequence.md
```

Notes:

- The parser accepts any `[...][...]` log prefix (`[DUQUANT][REPLACED]`, `[GR00T-DUQUANT][DRYRUN]`, etc.).
- Outputs a Markdown table with all layers (`idx`, module path, in→out) plus per-layer FP16/quant ms and speedup.
- Totals at the bottom report sequence runtime and aggregate speedup.

### Interpreting Results

- All timings are **mock** (random tensors) and include only kernel time measured via CUDA events.
- To profile real activations/weights, replace the mock tensor generation in each script with your data loader.
- INT4 activations (A4) require Tensor Core kernels, so the scripts automatically compile with a dynamic
  `M` range (includes 1, 8, 16, …) to keep decode-sized matmuls legal.
- If you need tuned kernels, flip `enable_tuning=True` when instantiating `Matmul`; be prepared for longer setup time.
