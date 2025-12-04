# DuQuant Quant Benchmark (gr00tN1.5_llm_ditmlp_layer.json)

| Layer (in→out) | Dtypes | Count | Batch | FP16 ms | Quant ms | Speedup |
| --- | --- | --- | --- | --- | --- | --- |
| 1536->6144 | W4A8 | 16 | 1 | 0.164088 | 0.116134 | 1.41× |
| 2048->1024 | W4A8 | 24 | 1 | 0.103982 | 0.115456 | 0.90× |
| 2048->2048 | W4A8 | 24 | 1 | 0.117155 | 0.115573 | 1.01× |
| 2048->6144 | W4A8 | 24 | 1 | 0.199253 | 0.124010 | 1.61× |
| 6144->1536 | W4A8 | 16 | 1 | 0.173621 | 0.116864 | 1.49× |
| 6144->2048 | W4A8 | 12 | 1 | 0.214029 | 0.117715 | 1.82× |

**FP16 total (count-weighted)**: 0.018 s
**Quant total (count-weighted)**: 0.014 s
**Count-weighted theoretical FLOPs**: 0.002 TFLOPs
**Overall weighted speedup**: 1.32×

_Totals multiply per-layer latency by occurrence count._