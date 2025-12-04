# DuQuant Quant Benchmark (pi_llm_ditmlp_layers_w4a4.json)

| Layer (in→out) | Dtypes | Count | Batch | FP16 ms | Quant ms | Speedup |
| --- | --- | --- | --- | --- | --- | --- |
| 1024->4096 | W4A4 | 36 | 1 | 0.107632 | 0.127728 | 0.84× |
| 2048->256 | W4A4 | 36 | 1 | 0.109110 | 0.128160 | 0.85× |
| 2048->2048 | W4A4 | 36 | 1 | 0.117875 | 0.131344 | 0.90× |
| 2048->16384 | W4A4 | 36 | 1 | 0.435411 | 0.184186 | 2.36× |
| 4096->1024 | W4A4 | 18 | 1 | 0.104358 | 0.134618 | 0.78× |
| 16384->2048 | W4A4 | 18 | 1 | 0.443507 | 0.218234 | 2.03× |

**FP16 total (count-weighted)**: 0.038 s
**Quant total (count-weighted)**: 0.027 s
**Count-weighted theoretical FLOPs**: 0.004 TFLOPs
**Overall weighted speedup**: 1.40×

_Totals multiply per-layer latency by occurrence count._