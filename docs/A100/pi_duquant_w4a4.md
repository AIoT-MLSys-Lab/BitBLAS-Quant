# DuQuant Quant Benchmark (pi_llm_ditmlp_layers_w4a4.json)

| Layer (in→out) | Dtypes | Count | Batch | FP16 ms | Quant ms | Speedup |
| --- | --- | --- | --- | --- | --- | --- |
| 1024->4096 | W4A4 | 36 | 10 | 0.039936 | 0.044288 | 0.90× |
| 2048->256 | W4A4 | 36 | 10 | 0.039680 | 0.042496 | 0.93× |
| 2048->2048 | W4A4 | 36 | 10 | 0.049408 | 0.037376 | 1.32× |
| 2048->16384 | W4A4 | 36 | 10 | 0.075008 | 0.040192 | 1.87× |
| 4096->1024 | W4A4 | 18 | 10 | 0.040704 | 0.085248 | 0.48× |
| 16384->2048 | W4A4 | 18 | 10 | 0.104960 | 0.068864 | 1.52× |

**FP16 total (count-weighted)**: 0.010 s
**Quant total (count-weighted)**: 0.009 s
**Count-weighted theoretical FLOPs**: 0.044 TFLOPs
**Overall weighted speedup**: 1.15×

_Totals multiply per-layer latency by occurrence count._