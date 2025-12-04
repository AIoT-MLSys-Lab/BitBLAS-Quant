# DuQuant Quant Benchmark (pi_llm_ditmlp_layers.json)

| Layer (in→out) | Dtypes | Count | Batch | FP16 ms | Quant ms | Speedup |
| --- | --- | --- | --- | --- | --- | --- |
| 1024->4096 | W4A8 | 36 | 1 | 0.026320 | 0.072803 | 0.36× |
| 2048->256 | W4A8 | 36 | 1 | 0.050061 | 0.037341 | 1.34× |
| 2048->2048 | W4A8 | 36 | 1 | 0.072253 | 0.075709 | 0.95× |
| 2048->16384 | W4A8 | 36 | 1 | 0.928138 | 0.359581 | 2.58× |
| 4096->1024 | W4A8 | 18 | 1 | 0.085923 | 0.075920 | 1.13× |
| 16384->2048 | W4A8 | 18 | 1 | 0.900534 | 0.365059 | 2.47× |

**FP16 total (count-weighted)**: 0.057 s
**Quant total (count-weighted)**: 0.028 s
**Count-weighted theoretical FLOPs**: 0.004 TFLOPs
**Overall weighted speedup**: 2.05×

_Totals multiply per-layer latency by occurrence count._