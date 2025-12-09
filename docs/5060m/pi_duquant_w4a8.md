# DuQuant Quant Benchmark (pi_llm_ditmlp_layers.json)

| Layer (in→out) | Dtypes | Count | Batch | FP16 ms | Quant ms | Speedup |
| --- | --- | --- | --- | --- | --- | --- |
| 1024->4096 | W4A8 | 36 | 1 | 0.019973 | 0.006098 | 3.28× |
| 2048->256 | W4A8 | 36 | 1 | 0.009699 | 0.005955 | 1.63× |
| 2048->2048 | W4A8 | 36 | 1 | 0.021229 | 0.006126 | 3.47× |
| 2048->16384 | W4A8 | 36 | 1 | 5.944573 | 0.006150 | 966.53× |
| 4096->1024 | W4A8 | 18 | 1 | 0.021333 | 0.006146 | 3.47× |
| 16384->2048 | W4A8 | 18 | 1 | 3.510354 | 0.005872 | 597.81× |

**FP16 total (count-weighted)**: 0.279 s
**Quant total (count-weighted)**: 0.001 s
**Count-weighted theoretical FLOPs**: 0.004 TFLOPs
**Overall weighted speedup**: 255.82×

_Totals multiply per-layer latency by occurrence count._