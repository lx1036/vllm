

# https://leetgpu.com/challenges/matrix-copy

import torch
from torch.utils.cpp_extension import load

# JIT 编译并加载 CUDA 扩展
lib = load(
    name="matrix_copy",
    # extra_include_paths=["include"],
    sources=["matrix_copy.cu"],
    verbose=True
)


N = 5
input = torch.full((N, N), 3.0, device='cuda')
output = torch.empty_like(input, device='cuda')
print(output)

lib.matrix_copy(input, output, N)
print(output)


# Loading extension module matrix_copy...
# tensor([[0., 0., 0., 0., 0.],
#         [0., 0., 0., 0., 0.],
#         [0., 0., 0., 0., 0.],
#         [0., 0., 0., 0., 0.],
#         [0., 0., 0., 0., 0.]], device='cuda:0')
# tensor([[3., 3., 3., 3., 3.],
#         [3., 3., 3., 3., 3.],
#         [3., 3., 3., 3., 3.],
#         [3., 3., 3., 3., 3.],
#         [3., 3., 3., 3., 3.]], device='cuda:0')


