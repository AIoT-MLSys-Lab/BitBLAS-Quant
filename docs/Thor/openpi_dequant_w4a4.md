# DuQuant Quant Benchmark (pi_llm_ditmlp_layers_w4a4.json)

| Layer (in→out) | Dtypes | Count | Batch | FP16 ms | Quant ms | Speedup |
| --- | --- | --- | --- | --- | --- | --- |
| 1024->4096 | W4A4 | 36 | 1 | 0.029864 | 0.102888 | 0.29× |
| 2048->256 | W4A4 | 36 | 1 | 0.046896 | 0.127080 | 0.37× |
| 2048->2048 | W4A4 | 36 | 1 | 0.072368 | 0.132792 | 0.54× |
| 2048->16384 | W4A4 | 36 | 1 | 0.948668 | 0.606368 | 1.56× |
| 4096->1024 | W4A4 | 18 | 1 | 0.092716 | 0.201276 | 0.46× |
| 16384->2048 | W4A4 | 18 | 1 | 0.903996 | 0.683944 | 1.32× |

**FP16 total (count-weighted)**: 0.057 s
**Quant total (count-weighted)**: 0.051 s
**Count-weighted theoretical FLOPs**: 0.004 TFLOPs
**Overall weighted speedup**: 1.13×

_Totals multiply per-layer latency by occurrence count._