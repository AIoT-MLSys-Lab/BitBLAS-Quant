# DuQuant Quant Benchmark (gr00tN1.5_llm_ditmlp_layer.json)

| Layer (in→out) | Dtypes | Count | Batch | FP16 ms | Quant ms | Speedup |
| --- | --- | --- | --- | --- | --- | --- |
| 1536->6144 | W4A8 | 16 | 1 | 0.032320 | 0.124992 | 0.26× |
| 2048->1024 | W4A8 | 24 | 1 | 0.059363 | 0.050080 | 1.19× |
| 2048->2048 | W4A8 | 24 | 1 | 0.070576 | 0.061098 | 1.16× |
| 2048->6144 | W4A8 | 24 | 1 | 0.146477 | 0.099296 | 1.48× |
| 6144->1536 | W4A8 | 16 | 1 | 0.121523 | 0.132531 | 0.92× |
| 6144->2048 | W4A8 | 12 | 1 | 0.149616 | 0.156787 | 0.95× |

**FP16 total (count-weighted)**: 0.011 s
**Quant total (count-weighted)**: 0.011 s
**Count-weighted theoretical FLOPs**: 0.002 TFLOPs
**Overall weighted speedup**: 0.99×

_Totals multiply per-layer latency by occurrence count._