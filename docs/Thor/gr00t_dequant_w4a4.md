# DuQuant Quant Benchmark (gr00tN1.5_llm_ditmlp_layer_w4a4.json)

| Layer (in→out) | Dtypes | Count | Batch | FP16 ms | Quant ms | Speedup |
| --- | --- | --- | --- | --- | --- | --- |
| 1536->6144 | W4A4 | 16 | 1 | 0.033069 | 0.181859 | 0.18× |
| 2048->1024 | W4A4 | 24 | 1 | 0.045155 | 0.116291 | 0.39× |
| 2048->2048 | W4A4 | 24 | 1 | 0.097027 | 0.119462 | 0.81× |
| 2048->6144 | W4A4 | 24 | 1 | 0.157267 | 0.221571 | 0.71× |
| 6144->1536 | W4A4 | 16 | 1 | 0.126221 | 0.288861 | 0.44× |
| 6144->2048 | W4A4 | 12 | 1 | 0.052342 | 0.299107 | 0.17× |

**FP16 total (count-weighted)**: 0.010 s
**Quant total (count-weighted)**: 0.022 s
**Count-weighted theoretical FLOPs**: 0.002 TFLOPs
**Overall weighted speedup**: 0.47×

_Totals multiply per-layer latency by occurrence count._