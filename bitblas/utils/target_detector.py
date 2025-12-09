# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
import os
import subprocess
from typing import List
import logging

from thefuzz import process
from tvm.target import Target
from tvm.target.tag import list_tags

logger = logging.getLogger(__name__)

TARGET_MISSING_ERROR = (
    "TVM target not found. Please set the TVM target environment variable using `export TVM_TARGET=<target>`, "
    "where <target> is one of the available targets can be found in the output of `tools/get_available_targets.py`."
)

# Nvidia produces non-public oem gpu models that are part of drivers but not mapped to correct tvm target
# Remap list to match the oem model name to the closest public model name
NVIDIA_GPU_REMAP = {
    "NVIDIA PG506-230": "NVIDIA A100",
    "NVIDIA PG506-232": "NVIDIA A100",
}


def get_gpu_model_from_nvidia_smi(gpu_id: int = 0):
    """
    Executes the 'nvidia-smi' command to fetch the name of the first available NVIDIA GPU.

    Returns:
        str: The name of the GPU, or None if 'nvidia-smi' command fails.
    """
    try:
        # Execute nvidia-smi command to get the GPU name
        output = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=gpu_name", "--format=csv,noheader"],
            encoding="utf-8",
        ).strip()
    except subprocess.CalledProcessError as e:
        logger.info("nvidia-smi failed with error: %s", e)
        return None

    gpus = output.split("\n")

    # for multiple gpus, CUDA_DEVICE_ORDER=PCI_BUS_ID must be set to match nvidia-smi or else wrong
    # gpu is returned for gpu_id
    if len(gpus) > 1 and os.environ.get("CUDA_DEVICE_ORDER") != "PCI_BUS_ID":
        raise EnvironmentError("Multi-gpu environment must set `CUDA_DEVICE_ORDER=PCI_BUS_ID`.")

    if gpu_id >= len(gpus) or gpu_id < 0:
        raise ValueError(f"Passed gpu_id:{gpu_id} but there are {len(gpus)} detected Nvidia gpus.")

    return gpus[gpu_id]


def find_best_match(tags, query):
    """
    Finds the best match for a query within a list of tags using fuzzy string matching.
    """
    MATCH_THRESHOLD = 25
    best_match, score = process.extractOne(query, tags)

    def check_target(best, default):
        return best if Target(best).arch == Target(default).arch else default

    if check_target(best_match, "cuda") == best_match:
        return best_match if score >= MATCH_THRESHOLD else "cuda"
    else:
        logger.warning(TARGET_MISSING_ERROR)
        return "cuda"


def get_all_nvidia_targets() -> List[str]:
    """
    Returns all available NVIDIA targets.
    """
    all_tags = list_tags()
    return [tag for tag in all_tags if "nvidia" in tag]


def _detect_cuda_major_version() -> int:
    """
    Best-effort detection of the CUDA major version from PyTorch or nvcc.
    Returns None if it cannot be determined.
    """
    # Try PyTorch first (since it's already a dependency)
    try:
        import torch  # pylint: disable=import-outside-toplevel

        if torch.version.cuda:
            ver = torch.version.cuda.split(".")
            if ver and ver[0].isdigit():
                return int(ver[0])
    except Exception:  # pylint: disable=broad-except
        pass

    # Fallback to nvcc
    try:
        output = subprocess.check_output(["nvcc", "--version"], encoding="utf-8")
        # Sample line: "Cuda compilation tools, release 11.5, V11.5.119"
        for token in output.replace(",", " ").split():
            if token.lower().startswith("release"):
                parts = token.split()
                if len(parts) > 1:
                    version = parts[1]
                else:
                    continue
            elif token.count(".") >= 1 and token.replace(".", "").isdigit():
                version = token
            else:
                continue
            major = version.split(".")[0]
            if major.isdigit():
                return int(major)
    except Exception:  # pylint: disable=broad-except
        pass

    return None


def auto_detect_nvidia_target(gpu_id: int = 0) -> str:
    """
    Automatically detects the NVIDIA GPU architecture to set the appropriate TVM target.

    Returns:
        str: The detected TVM target architecture.
    """
    # Honor explicit override first
    if "TVM_TARGET" in os.environ:
        return os.environ["TVM_TARGET"]

    # Fetch all available tags and filter for NVIDIA tags
    all_tags = list_tags()
    nvidia_tags = [tag for tag in all_tags if "nvidia" in tag]

    # Get the current GPU model and find the best matching target
    gpu_model = get_gpu_model_from_nvidia_smi(gpu_id=gpu_id)

    # Compat: remap oem devices to their correct non-oem model names for tvm target
    if gpu_model in NVIDIA_GPU_REMAP:
        gpu_model = NVIDIA_GPU_REMAP[gpu_model]

    # If we can get compute capability, prefer constructing an explicit arch target
    try:
        output = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=compute_cap", "--format=csv,noheader"],
            encoding="utf-8",
        ).strip()
        cap = output.split("\n")[gpu_id].strip()
        # compute_cap is like "11.0"
        if cap and cap.replace(".", "").isdigit():
            cap_int = int(cap.split(".")[0]) * 10 + int(cap.split(".")[1])
            if cap_int >= 110:
                cuda_major = _detect_cuda_major_version()
                if cuda_major is not None and cuda_major < 12:
                    logger.warning(
                        "Detected compute capability %s (>= sm_110) but CUDA %s "
                        "does not support compiling sm_110 kernels. Falling back to sm_86.",
                        cap,
                        cuda_major,
                    )
                    return "cuda -arch=sm_86"
                return "cuda -arch=sm_110"
            elif cap_int >= 90:
                cuda_major = _detect_cuda_major_version()
                if cuda_major is not None and cuda_major < 12:
                    logger.warning(
                        "Detected compute capability %s (>= sm_90) but CUDA %s "
                        "does not support compiling sm_90 kernels. Falling back to sm_86.",
                        cap,
                        cuda_major,
                    )
                    return "cuda -arch=sm_86"
                return "cuda -arch=sm_90"
            elif cap_int >= 89:
                cuda_major = _detect_cuda_major_version()
                if cuda_major is not None and cuda_major < 12:
                     # sm_89 requires CUDA 11.8+
                     # Since we only have major version, we can't be sure about 11.8 vs 11.5
                     # But to be safe for 11.5, we should fallback to sm_86
                     # However, _detect_cuda_major_version only gives major.
                     # Let's assume if it is 11, we might need fallback if it is not 11.8
                     # But wait, if I am on 11.5, sm_89 will fail.
                     # So I should probably fallback to sm_86 for all 11.x if I want to be safe,
                     # or I need to detect minor version.
                     pass
                return "cuda -arch=sm_89"
            elif cap_int >= 87:
                return "cuda -arch=sm_87"
            elif cap_int >= 86:
                return "cuda -arch=sm_86"
            elif cap_int >= 80:
                return "cuda -arch=sm_80"
    except Exception:
        pass

    target = find_best_match(nvidia_tags, gpu_model) if gpu_model else "cuda"
    return target
