# DuQuant Quant Benchmark (gr00tN1.5_llm_ditmlp_layer_w4a4.json)

| Layer (in→out) | Dtypes | Count | Batch | FP16 ms | Quant ms | Speedup |
| --- | --- | --- | --- | --- | --- | --- |
| 1536->6144 | W4A4 | 16 | 10 | 0.047616 | 0.038144 | 1.25× |
| 2048->1024 | W4A4 | 24 | 10 | 0.040448 | 0.036864 | 1.10× |
| 2048->2048 | W4A4 | 24 | 10 | 0.039936 | 0.037376 | 1.07× |
| 2048->6144 | W4A4 | 24 | 10 | 0.037376 | 0.041984 | 0.89× |
| 6144->1536 | W4A4 | 16 | 10 | 0.041728 | 0.044032 | 0.95× |
| 6144->2048 | W4A4 | 12 | 10 | 0.040960 | 0.044032 | 0.93× |

**FP16 total (count-weighted)**: 0.005 s
**Quant total (count-weighted)**: 0.005 s
**Count-weighted theoretical FLOPs**: 0.018 TFLOPs
**Overall weighted speedup**: 1.02×

_Totals multiply per-layer latency by occurrence count._