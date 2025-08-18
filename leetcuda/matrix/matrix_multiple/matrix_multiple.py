

# https://leetgpu.com/challenges/matrix-multiplication

import torch
from torch.utils.cpp_extension import load

# JIT 编译并加载 CUDA 扩展
lib = load(
    name="matrix_multiple",
    # extra_include_paths=["include"],
    sources=["matrix_multiple.cu"],
    verbose=True
)


# 定义矩阵的维度
M = 10
N = 5
K = 8
# # 创建矩阵 A (MxN)
# A = torch.randn(M, N) # 10*5
# # 创建矩阵 B (NxK)
# B = torch.randn(N, K) # 5*8
# C = torch.zeros(M, K) # 10*8

A = torch.full((M, N), 3.0, device='cuda')
B = torch.full((N, K), 3.0, device='cuda')
C = torch.full((M, K), 0.0, device='cuda')

out = torch.matmul(A, B)
# print(torch.allclose(out, C))
print(out)

lib.matrix_multiple(A, B, C, M, N, K)
print(C)

if torch.equal(out, C):
    print("Success")
else:
    print("Failed")


# Loading extension module matrix_multiple...
# tensor([[45., 45., 45., 45., 45., 45., 45., 45.],
#         [45., 45., 45., 45., 45., 45., 45., 45.],
#         [45., 45., 45., 45., 45., 45., 45., 45.],
#         [45., 45., 45., 45., 45., 45., 45., 45.],
#         [45., 45., 45., 45., 45., 45., 45., 45.],
#         [45., 45., 45., 45., 45., 45., 45., 45.],
#         [45., 45., 45., 45., 45., 45., 45., 45.],
#         [45., 45., 45., 45., 45., 45., 45., 45.],
#         [45., 45., 45., 45., 45., 45., 45., 45.],
#         [45., 45., 45., 45., 45., 45., 45., 45.]], device='cuda:0')
# tensor([[45., 45., 45., 45., 45., 45., 45., 45.],
#         [45., 45., 45., 45., 45., 45., 45., 45.],
#         [45., 45., 45., 45., 45., 45., 45., 45.],
#         [45., 45., 45., 45., 45., 45., 45., 45.],
#         [45., 45., 45., 45., 45., 45., 45., 45.],
#         [45., 45., 45., 45., 45., 45., 45., 45.],
#         [45., 45., 45., 45., 45., 45., 45., 45.],
#         [45., 45., 45., 45., 45., 45., 45., 45.],
#         [45., 45., 45., 45., 45., 45., 45., 45.],
#         [45., 45., 45., 45., 45., 45., 45., 45.]], device='cuda:0')
# Success
