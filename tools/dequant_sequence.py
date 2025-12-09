#!/usr/bin/env python3
"""Sequential mock benchmark for DuQuant layer logs (supports W/A dtype parsing)."""

from __future__ import annotations

import argparse
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import torch

from bitblas import Matmul, MatmulConfig, auto_detect_nvidia_target


@dataclass
class LayerEntry:
    idx: int
    name: str
    in_features: int
    out_features: int
    w_dtype: str
    a_dtype: str

    @property
    def shape_str(self) -> str:
        return f"{self.in_features}->{self.out_features}"

    @property
    def wa_label(self) -> str:
        return f"W{dtype_bits(self.w_dtype)}A{dtype_bits(self.a_dtype)}"


def parse_layer_sequence(path: Path) -> List[LayerEntry]:
    base_pattern = re.compile(r"Linear\((\d+)->(\d+)\)")
    wa_pattern = re.compile(r"W\s*([A-Za-z0-9]+)\s+A\s*([A-Za-z0-9]+)")
    entries: List[LayerEntry] = []
    with path.open() as fin:
        for line in fin:
            base = base_pattern.search(line)
            if not base:
                continue
            w_token, a_token = "4", "8"
            wa_match = wa_pattern.search(line)
            if wa_match:
                w_token, a_token = wa_match.groups()
            entries.append(
                LayerEntry(
                    idx=len(entries),
                    name=line.split(":")[0].split(" ", 1)[-1].strip(),
                    in_features=int(base.group(1)),
                    out_features=int(base.group(2)),
                    w_dtype=normalize_duquant_token(w_token, is_weight=True),
                    a_dtype=normalize_duquant_token(a_token, is_weight=False),
                )
            )
    if not entries:
        raise ValueError(f"No DUQUANT Linear entries found in {path}")
    return entries


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
    matmul: Matmul,
    a_dtype: str,
    w_dtype: str,
) -> float:
    activation = sample_int_tensor((M, K), a_dtype)
    weight = sample_int_tensor((N, K), w_dtype)
    packed_weight = matmul.transform_weight(weight)

    def op():
        return matmul(activation, packed_weight)

    # Validation step
    output = op()
    if torch.all(output == 0):
        raise RuntimeError(
            f"Benchmark validation failed: Output is all zeros for shape M={M}, N={N}, K={K}. "
            "This indicates the kernel is not executing correctly (likely architecture mismatch)."
        )

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


def get_bitblas_matmul(
    cache: Dict[Tuple[int, int, str, str], Matmul],
    *,
    M: int,
    N: int,
    K: int,
    a_dtype: str,
    w_dtype: str,
    target: str,
) -> Matmul:
    key = (N, K, a_dtype, w_dtype)
    if key not in cache:
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
        cache[key] = Matmul(config, target=target, enable_tuning=False)
    return cache[key]


def write_markdown(entries: List[Dict], output: Path, title: str):
    total_fp16 = sum(item["fp16_ms"] for item in entries)
    total_quant = sum(item["quant_ms"] for item in entries)
    speedup = total_fp16 / total_quant

    lines = [
        f"# {title}",
        "",
        "| # | Layer | Shape | Dtypes | FP16 ms | Quant ms | Speedup |",
        "| --- | --- | --- | --- | --- | --- | --- |",
    ]
    for item in entries:
        lines.append(
            f"| {item['idx']} | `{item['name']}` | {item['shape']} | {item['wa_label']} | "
            f"{item['fp16_ms']:.6f} | {item['quant_ms']:.6f} | {item['speedup']:.2f}× |"
        )
    lines.extend(
        [
            "",
            f"**Total FP16 time (sequence)**: {total_fp16 / 1e3:.3f} s",
            f"**Total Quant time (sequence)**: {total_quant / 1e3:.3f} s",
            f"**Overall sequential speedup**: {speedup:.2f}×",
        ]
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text("\n".join(lines))
    print(f"Saved sequential report to {output}")
    print(f"Sequential speedup: {speedup:.2f}×")


def normalize_duquant_token(token: str, *, is_weight: bool) -> str:
    token = token.strip().lower()
    if token.startswith("int") or token.startswith("uint"):
        return token
    if token.startswith("u"):
        return f"uint{int(token[1:])}"
    if token.startswith("a") or token.startswith("w"):
        token = token[1:]
    digits = "".join(ch for ch in token if ch.isdigit())
    return f"int{digits or '8'}" if is_weight else f"int{digits or '8'}"


def dtype_bits(dtype: str) -> int:
    digits = "".join(ch for ch in dtype if ch.isdigit())
    return int(digits) if digits else 0


def dtype_is_unsigned(dtype: str) -> bool:
    return dtype.startswith("uint")


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
    parser = argparse.ArgumentParser(description="Sequential DuQuant benchmark")
    parser.add_argument(
        "--layer-log",
        type=Path,
        default=Path("pi_llm_ditmlp_layers.json"),
        help="File that lists Linear replacements in order.",
    )
    parser.add_argument("--batch", type=int, default=1, help="Batch size M.")
    parser.add_argument("--warmup", type=int, default=2, help="Warmup iterations per layer.")
    parser.add_argument("--runs", type=int, default=8, help="Measured iterations per layer.")
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("reports/pi_duquant_w4a8_sequence.md"),
        help="Markdown output path.",
    )
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    sequence = parse_layer_sequence(args.layer_log)
    target = auto_detect_nvidia_target()
    matmul_cache: Dict[Tuple[int, int, str, str], Matmul] = {}

    results: List[Dict] = []
    for entry in sequence:
        print(f"[{entry.idx:03d}] {entry.name} ({entry.shape_str}, {entry.wa_label})")
        fp16_ms = benchmark_fp16(args.batch, entry.out_features, entry.in_features,
                                 args.warmup, args.runs)
        matmul = get_bitblas_matmul(
            matmul_cache,
            M=args.batch,
            N=entry.out_features,
            K=entry.in_features,
            a_dtype=entry.a_dtype,
            w_dtype=entry.w_dtype,
            target=target,
        )
        quant_ms = benchmark_quantized(args.batch, entry.out_features, entry.in_features,
                                       args.warmup, args.runs, matmul, entry.a_dtype,
                                       entry.w_dtype)
        results.append(
            {
                "idx": entry.idx,
                "name": entry.name,
                "shape": entry.shape_str,
                "wa_label": entry.wa_label,
                "fp16_ms": fp16_ms,
                "quant_ms": quant_ms,
                "speedup": fp16_ms / quant_ms if quant_ms > 0 else float("inf"),
            }
        )

    title = f"DuQuant Sequential Benchmark ({args.layer_log.name})"
    write_markdown(results, args.output, title)


if __name__ == "__main__":
    main()
