#!/usr/bin/env python3
"""Benchmark DuQuant mock layers (shape-collapsed) for arbitrary W/A dtypes."""

from __future__ import annotations

import argparse
import re
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import torch

from bitblas import Matmul, MatmulConfig, auto_detect_nvidia_target


@dataclass
class LayerShape:
    in_features: int
    out_features: int
    count: int
    batch: int
    w_dtype: str
    a_dtype: str

    @property
    def name(self) -> str:
        return f"{self.in_features}->{self.out_features}"

    @property
    def wa_label(self) -> str:
        return f"W{dtype_bits(self.w_dtype)}A{dtype_bits(self.a_dtype)}"


def parse_layer_log(path: Path, batch: int) -> List[LayerShape]:
    base_pattern = re.compile(r"Linear\((\d+)->(\d+)\)")
    wa_pattern = re.compile(r"W\s*([A-Za-z0-9]+)\s+A\s*([A-Za-z0-9]+)")
    counts: Counter[Tuple[int, int, str, str]] = Counter()
    with path.open() as fin:
        for line in fin:
            base = base_pattern.search(line)
            if not base:
                continue
            w_token, a_token = "4", "8"  # defaults
            wa_match = wa_pattern.search(line)
            if wa_match:
                w_token, a_token = wa_match.groups()
            w_dtype = normalize_duquant_token(w_token, is_weight=True)
            a_dtype = normalize_duquant_token(a_token, is_weight=False)
            key = (int(base.group(1)), int(base.group(2)), w_dtype, a_dtype)
            counts[key] += 1
    if not counts:
        raise ValueError(f"No Linear entries found in {path}")
    return [
        LayerShape(
            in_features=fin,
            out_features=fout,
            count=count,
            batch=batch,
            w_dtype=w_dtype,
            a_dtype=a_dtype,
        )
        for (fin, fout, w_dtype, a_dtype), count in sorted(counts.items())
    ]


def benchmark_fp16(M: int, N: int, K: int, warmup: int, runs: int) -> float:
    activation = torch.randn((M, K), device="cuda", dtype=torch.float16)
    weight = torch.randn((N, K), device="cuda", dtype=torch.float16)

    def op():
        return torch.matmul(activation, weight.t())

    return _time_cuda_op(op, warmup, runs)


def benchmark_quantized(
    M: int,
    N: int,
    K: int,
    warmup: int,
    runs: int,
    *,
    a_dtype: str,
    w_dtype: str,
    target: str,
    tune: bool = False,
    tune_topk: int = 10,
) -> float:
    config = MatmulConfig(
        M=legalize_m_dim(M, a_dtype),
        N=N,
        K=K,
        A_dtype=a_dtype,
        W_dtype=w_dtype,
        accum_dtype="int32",
        out_dtype="float16",
        layout="nt",
        with_bias=False,
        group_size=None,
        with_scaling=False,
        with_zeros=False,
        zeros_mode=None,
    )
    matmul = Matmul(config, target=target, enable_tuning=False)

    if tune:
        matmul.hardware_aware_finetune(topk=tune_topk, parallel_build=True)

    activation = sample_int_tensor((M, K), a_dtype)
    weight = sample_int_tensor((N, K), w_dtype)
    packed_weight = matmul.transform_weight(weight)

    def op():
        return matmul(activation, packed_weight)

    return _time_cuda_op(op, warmup, runs)


def _time_cuda_op(fn, warmup: int, runs: int) -> float:
    torch.cuda.synchronize()
    for _ in range(max(0, warmup)):
        fn()
    torch.cuda.synchronize()

    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    samples: List[float] = []
    for _ in range(runs):
        start.record()
        fn()
        end.record()
        end.synchronize()
        samples.append(start.elapsed_time(end))
    return sum(samples) / len(samples)


def save_markdown(results: List[Dict], output: Path, title: str):
    total_fp16 = sum(item["fp16_ms"] * item["count"] for item in results)
    total_quant = sum(item["quant_ms"] * item["count"] for item in results)
    total_flops = sum(item["flops"] for item in results)
    overall_speedup = total_fp16 / total_quant

    lines = [
        f"# {title}",
        "",
        "| Layer (in→out) | Dtypes | Count | Batch | FP16 ms | Quant ms | Speedup |",
        "| --- | --- | --- | --- | --- | --- | --- |",
    ]
    for item in results:
        lines.append(
            f"| {item['shape']} | {item['wa_label']} | {item['count']} | {item['batch']} | "
            f"{item['fp16_ms']:.6f} | {item['quant_ms']:.6f} | {item['speedup']:.2f}× |"
        )
    lines.extend(
        [
            "",
            f"**FP16 total (count-weighted)**: {total_fp16 / 1e3:.3f} s",
            f"**Quant total (count-weighted)**: {total_quant / 1e3:.3f} s",
            f"**Count-weighted theoretical FLOPs**: {total_flops / 1e12:.3f} TFLOPs",
            f"**Overall weighted speedup**: {overall_speedup:.2f}×",
            "",
            "_Totals multiply per-layer latency by occurrence count._",
        ]
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text("\n".join(lines))
    print(f"Saved Markdown report to {output}")
    print(f"Weighted speedup: {overall_speedup:.2f}×")


def normalize_duquant_token(token: str, *, is_weight: bool) -> str:
    token = token.strip()
    token = token.lower()
    prefixes = ("int", "uint", "fp", "nf")
    for prefix in prefixes:
        if token.startswith(prefix):
            suffix = token[len(prefix):]
            return f"{prefix}{suffix}" if prefix != "fp" else token
    if token.startswith("a") or token.startswith("w"):
        token = token[1:]
    # default to signed integer
    if not token:
        return "int8"
    return f"int{int(token)}" if is_weight else f"int{int(token)}"


def dtype_bits(dtype: str) -> int:
    digits = "".join(ch for ch in dtype if ch.isdigit())
    return int(digits) if digits else 0


def dtype_is_unsigned(dtype: str) -> bool:
    return dtype.startswith("uint") or dtype.startswith("u")


def sample_int_tensor(shape: Tuple[int, int], dtype: str) -> torch.Tensor:
    bits = dtype_bits(dtype)
    signed = not dtype_is_unsigned(dtype)
    if bits == 0:
        raise ValueError(f"Unsupported dtype {dtype}")
    if signed:
        low, high = -(2 ** (bits - 1)), 2 ** (bits - 1)
    else:
        low, high = 0, 2 ** bits
    return torch.randint(low, high, shape, device="cuda", dtype=torch.int8)


def legalize_m_dim(batch: int, a_dtype: str):
    if a_dtype in ("int4", "uint4") and isinstance(batch, int) and batch < 8:
        candidates = sorted({batch, 8, 16, 32, 64, 128, 256, 512, 1024})
        return tuple(candidates)
    return batch


def main():
    parser = argparse.ArgumentParser(description="Benchmark DuQuant vs FP16 (shape dedup)")
    parser.add_argument(
        "--layer-log",
        type=Path,
        default=Path("pi_llm_ditmlp_layers.json"),
        help="Path to DuQuant layer log file.",
    )
    parser.add_argument("--batch", type=int, default=1, help="Batch/sequence size M.")
    parser.add_argument("--warmup", type=int, default=10, help="Warmup iterations.")
    parser.add_argument("--runs", type=int, default=50, help="Measured iterations.")
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("reports/pi_duquant_w4a8.md"),
        help="Markdown output path.",
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--tune", action="store_true", help="Run hardware-aware tuning for each shape.")
    parser.add_argument("--tune-topk", type=int, default=10, help="Top-K configs for tuning.")
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    shapes = parse_layer_log(args.layer_log, args.batch)
    target = auto_detect_nvidia_target()

    results = []
    for shape in shapes:
        print(
            f"Benchmarking {shape.name} ({shape.wa_label}) count={shape.count}"
        )
        fp16_ms = benchmark_fp16(shape.batch, shape.out_features, shape.in_features,
                                 args.warmup, args.runs)
        quant_ms = benchmark_quantized(
            shape.batch,
            shape.out_features,
            shape.in_features,
            args.warmup,
            args.runs,
            a_dtype=shape.a_dtype,
            w_dtype=shape.w_dtype,
            target=target,
            tune=args.tune,
            tune_topk=args.tune_topk,
        )
        results.append(
            {
                "shape": shape.name,
                "wa_label": shape.wa_label,
                "count": shape.count,
                "batch": shape.batch,
                "fp16_ms": fp16_ms,
                "quant_ms": quant_ms,
                "speedup": fp16_ms / quant_ms if quant_ms > 0 else float("inf"),
                "flops": 2
                * shape.count
                * shape.batch
                * shape.out_features
                * shape.in_features,
            }
        )

    title = f"DuQuant Quant Benchmark ({args.layer_log.name})"
    save_markdown(results, args.output, title)


if __name__ == "__main__":
    main()
