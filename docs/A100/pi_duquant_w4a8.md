# DuQuant Quant Benchmark (pi_llm_ditmlp_layers.json)

| Layer (in→out) | Dtypes | Count | Batch | FP16 ms | Quant ms | Speedup |
| --- | --- | --- | --- | --- | --- | --- |
| 1024->4096 | W4A8 | 36 | 10 | 0.042496 | 0.051968 | 0.82× |
| 2048->256 | W4A8 | 36 | 10 | 0.041472 | 0.071168 | 0.58× |
| 2048->2048 | W4A8 | 36 | 10 | 0.048896 | 0.068352 | 0.72× |
| 2048->16384 | W4A8 | 36 | 10 | 0.075008 | 0.093696 | 0.80× |
| 4096->1024 | W4A8 | 18 | 10 | 0.043776 | 0.106752 | 0.41× |
| 16384->2048 | W4A8 | 18 | 10 | 0.072192 | 0.348160 | 0.21× |

**FP16 total (count-weighted)**: 0.010 s
**Quant total (count-weighted)**: 0.018 s
**Count-weighted theoretical FLOPs**: 0.044 TFLOPs
**Overall weighted speedup**: 0.52×

_Totals multiply per-layer latency by occurrence count._