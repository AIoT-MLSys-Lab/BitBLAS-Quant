
import torch
from bitblas import Matmul, MatmulConfig
import sys

def benchmark_quantized(M, N, K, a_dtype, w_dtype, target):
    print(f"Benchmarking M={M}, N={N}, K={K}, A={a_dtype}, W={w_dtype}, target={target}")
    
    # legalize_m_dim logic inline
    if a_dtype in ("int4", "uint4") and isinstance(M, int) and M < 8:
        M = tuple(sorted({M, 8, 16, 32, 64, 128, 256, 512, 1024}))
    
    config = MatmulConfig(
        M=M,
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
    print(f"DEBUG: Config: {config}")
    sys.stdout.flush()
    
    matmul = Matmul(config, target=target, enable_tuning=False)

    # sample_int_tensor logic inline
    def sample(shape, dtype):
        bits = int("".join(ch for ch in dtype if ch.isdigit()))
        signed = not (dtype.startswith("uint") or dtype.startswith("u"))
        if signed:
            low, high = -(2 ** (bits - 1)), 2 ** (bits - 1)
        else:
            low, high = 0, 2 ** bits
        return torch.randint(low, high, shape, device="cuda", dtype=torch.int8)

    activation = sample((1, K), a_dtype) # M is 1 here for tensor shape
    weight = sample((N, K), w_dtype)
    
    packed_weight = matmul.transform_weight(weight)

    def op():
        return matmul(activation, packed_weight)

    # Validation step
    output = op()
    print(f"DEBUG: Output stats: min={output.min()}, max={output.max()}, mean={output.float().mean()}")
    
    if torch.all(output == 0):
        print("FAILURE: Output is all zeros")
    else:
        print("SUCCESS: Output is non-zero")

if __name__ == "__main__":
    benchmark_quantized(1, 256, 2048, "int8", "int4", "cuda -arch=sm_100")
