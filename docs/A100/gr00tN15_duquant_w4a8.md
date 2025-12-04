# DuQuant Quant Benchmark (gr00tN1.5_llm_ditmlp_layer.json)

| Layer (in→out) | Dtypes | Count | Batch | FP16 ms | Quant ms | Speedup |
| --- | --- | --- | --- | --- | --- | --- |
| 1536->6144 | W4A8 | 16 | 10 | 0.042752 | 0.059136 | 0.72× |
| 2048->1024 | W4A8 | 24 | 10 | 0.040960 | 0.066560 | 0.62× |
| 2048->2048 | W4A8 | 24 | 10 | 0.038656 | 0.068096 | 0.57× |
| 2048->6144 | W4A8 | 24 | 10 | 0.042496 | 0.068096 | 0.62× |
| 6144->1536 | W4A8 | 16 | 10 | 0.041984 | 0.147712 | 0.28× |
| 6144->2048 | W4A8 | 12 | 10 | 0.042496 | 0.147456 | 0.29× |

**FP16 total (count-weighted)**: 0.005 s
**Quant total (count-weighted)**: 0.010 s
**Count-weighted theoretical FLOPs**: 0.018 TFLOPs
**Overall weighted speedup**: 0.48×

_Totals multiply per-layer latency by occurrence count._