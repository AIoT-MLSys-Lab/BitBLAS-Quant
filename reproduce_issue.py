
import torch
import sys
from bitblas import Matmul, MatmulConfig

def log(msg):
    print(msg)
    sys.stdout.flush()

def benchmark_fp16(M, N, K):
    log("Running FP16 benchmark...")
    activation = torch.randn((M, K), device="cuda", dtype=torch.float16)
    weight = torch.randn((N, K), device="cuda", dtype=torch.float16)
    def op():
        return torch.matmul(activation, weight.t())
    
    # Warmup
    for _ in range(5):
        op()
    torch.cuda.synchronize()
    log("FP16 benchmark done.")

def verify_layer(M, N, K, w_dtype, a_dtype):
    log(f"Verifying {M}x{N}x{K} W={w_dtype} A={a_dtype}")
    
    config = MatmulConfig(
        M=M, N=N, K=K,
        A_dtype=a_dtype, W_dtype=w_dtype,
        accum_dtype="int32", out_dtype="float16",
        layout="nt", with_bias=False,
        group_size=None, with_scaling=False, with_zeros=False
    )
    
    target = "cuda -arch=sm_100"
    log(f"Target: {target}")
    
    try:
        matmul = Matmul(config, target=target, enable_tuning=False)
        log("Matmul created")
    except Exception as e:
        log(f"Matmul creation failed: {e}")
        return False

    A = torch.randint(-128, 128, (M, K), dtype=torch.int8).cuda()
    W = torch.randint(-8, 8, (N, K), dtype=torch.int8).cuda()
    
    try:
        packed_weight = matmul.transform_weight(W)
        log("Weight transformed")
        
        output = matmul(A, packed_weight)
        log("Forward pass done")
    except Exception as e:
        log(f"Execution failed: {e}")
        return False
    
    log(f"Output max: {output.max().item()}")
    
    if torch.all(output == 0):
        log("ERROR: Output is all zeros!")
        return False
        
    log("SUCCESS: Output is non-zero")
    return True

if __name__ == "__main__":
    # 1. Run FP16 benchmark (like dequant_benchmark.py)
    # benchmark_fp16(1, 256, 2048)
    
    # 2. Run Quantized verification
    verify_layer(M=1, K=2048, N=256, w_dtype="int4", a_dtype="int8")
