# DuQuant Quant Benchmark (pi_llm_ditmlp_layers.json)

| Layer (in→out) | Dtypes | Count | Batch | FP16 ms | Quant ms | Speedup |
| --- | --- | --- | --- | --- | --- | --- |
| 1024->4096 | W4A8 | 36 | 1 | 0.101994 | 0.130534 | 0.78× |
| 2048->256 | W4A8 | 36 | 1 | 0.111853 | 0.127706 | 0.88× |
| 2048->2048 | W4A8 | 36 | 1 | 0.124579 | 0.133418 | 0.93× |
| 2048->16384 | W4A8 | 36 | 1 | 0.436262 | 0.197763 | 2.21× |
| 4096->1024 | W4A8 | 18 | 1 | 0.106867 | 0.130102 | 0.82× |
| 16384->2048 | W4A8 | 18 | 1 | 0.444214 | 0.187734 | 2.37× |

**FP16 total (count-weighted)**: 0.038 s
**Quant total (count-weighted)**: 0.027 s
**Count-weighted theoretical FLOPs**: 0.004 TFLOPs
**Overall weighted speedup**: 1.40×

_Totals multiply per-layer latency by occurrence count._