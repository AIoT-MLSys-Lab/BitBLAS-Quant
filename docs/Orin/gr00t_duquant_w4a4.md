# DuQuant Quant Benchmark (gr00tN1.5_llm_ditmlp_layer_w4a4.json)

| Layer (in→out) | Dtypes | Count | Batch | FP16 ms | Quant ms | Speedup |
| --- | --- | --- | --- | --- | --- | --- |
| 1536->6144 | W4A4 | 16 | 1 | 0.162936 | 0.130144 | 1.25× |
| 2048->1024 | W4A4 | 24 | 1 | 0.104166 | 0.126570 | 0.82× |
| 2048->2048 | W4A4 | 24 | 1 | 0.116443 | 0.128349 | 0.91× |
| 2048->6144 | W4A4 | 24 | 1 | 0.198987 | 0.127989 | 1.55× |
| 6144->1536 | W4A4 | 16 | 1 | 0.173077 | 0.126541 | 1.37× |
| 6144->2048 | W4A4 | 12 | 1 | 0.212237 | 0.128122 | 1.66× |

**FP16 total (count-weighted)**: 0.018 s
**Quant total (count-weighted)**: 0.015 s
**Count-weighted theoretical FLOPs**: 0.002 TFLOPs
**Overall weighted speedup**: 1.21×

_Totals multiply per-layer latency by occurrence count._