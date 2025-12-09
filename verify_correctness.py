
import torch
import sys
from bitblas import Matmul, MatmulConfig, auto_detect_nvidia_target

def log(msg):
    print(msg)
    sys.stdout.flush()

def verify_layer(M, N, K, w_dtype, a_dtype):
    # Failing shape from log: M=1, N=256, K=2048
    M, N, K = 1, 256, 2048
    log(f"Verifying {M}x{N}x{K} W={w_dtype} A={a_dtype}")
    
    # Use default config to match benchmark behavior
    config = MatmulConfig(
        M=M, N=N, K=K,
        A_dtype=a_dtype, W_dtype=w_dtype,
        accum_dtype="int32", out_dtype="float16",
        layout="nt", with_bias=False,
        group_size=None, with_scaling=False, with_zeros=False
    )
    log(f"DEBUG: Config: {config}")
    
    target = "cuda -arch=sm_100" # Back to sm_100
    log(f"Target: {target}")
    
    try:
        matmul = Matmul(config, target=target, enable_tuning=False)
        log("Matmul created")
    except Exception as e:
        log(f"Matmul creation failed: {e}")
        return False

    # Use full range inputs to match benchmark
    # dequant_benchmark uses sample_int_tensor: low, high = -(2**(bits-1)), 2**(bits-1)
    # torch.randint is [low, high)
    A = torch.randint(-128, 128, (M, K), dtype=torch.int8).cuda()
    W = torch.randint(-8, 8, (N, K), dtype=torch.int8).cuda()
    
    try:
        packed_weight = matmul.transform_weight(W)
        log("Weight transformed")
        
        log(f"A stats: min={A.min()}, max={A.max()}, mean={A.float().mean()}")
        log(f"W stats: min={W.min()}, max={W.max()}, mean={W.float().mean()}")
        log(f"Packed W stats: min={packed_weight.min()}, max={packed_weight.max()}, mean={packed_weight.float().mean()}")
        if torch.all(packed_weight == 0):
             log("ERROR: Packed weight is all zeros!")
        
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
    verify_layer(M=1, K=1024, N=4096, w_dtype="int4", a_dtype="int8")
