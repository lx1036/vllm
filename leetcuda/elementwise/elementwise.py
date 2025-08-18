

# https://github.com/xlite-dev/LeetCUDA/blob/main/kernels/elementwise/elementwise.py


import torch
from torch.utils.cpp_extension import load


# Load the CUDA kernel as a python module
lib = load(
    name="elementwise_lib",
    sources=["elementwise.cu"],
    extra_cuda_cflags=[
        "-O3",
        "-U__CUDA_NO_HALF_OPERATORS__",
        "-U__CUDA_NO_HALF_CONVERSIONS__",
        "-U__CUDA_NO_HALF2_OPERATORS__",
        "-U__CUDA_NO_BFLOAT16_CONVERSIONS__",
        "--expt-relaxed-constexpr",
        "--expt-extended-lambda",
        "--use_fast_math",
    ],
    extra_cflags=["-std=c++17"],
)


import torch

# A, B, C are tensors on the GPU
def solve(A: torch.Tensor, B: torch.Tensor, C: torch.Tensor, N: int):
    torch.add(A, B, out=C)
