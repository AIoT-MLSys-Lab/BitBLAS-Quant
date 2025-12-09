# DuQuant Quant Benchmark (gr00tN1.5_llm_ditmlp_layer.json)

| Layer (in→out) | Dtypes | Count | Batch | FP16 ms | Quant ms | Speedup |
| --- | --- | --- | --- | --- | --- | --- |
| 1536->6144 | W4A8 | 16 | 1 | 0.041264 | 0.005928 | 6.96× |
| 2048->1024 | W4A8 | 24 | 1 | 0.015091 | 0.005878 | 2.57× |
| 2048->2048 | W4A8 | 24 | 1 | 0.021658 | 0.006256 | 3.46× |
| 2048->6144 | W4A8 | 24 | 1 | 0.341552 | 0.006038 | 56.56× |
| 6144->1536 | W4A8 | 16 | 1 | 0.039459 | 0.006125 | 6.44× |
| 6144->2048 | W4A8 | 12 | 1 | 0.114430 | 0.005925 | 19.31× |

**FP16 total (count-weighted)**: 0.012 s
**Quant total (count-weighted)**: 0.001 s
**Count-weighted theoretical FLOPs**: 0.002 TFLOPs
**Overall weighted speedup**: 16.77×

_Totals multiply per-layer latency by occurrence count._