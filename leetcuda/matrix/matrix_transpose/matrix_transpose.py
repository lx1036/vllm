

# https://leetgpu.com/challenges/matrix-transpose


import torch
from torch.utils.cpp_extension import load

# JIT 编译并加载 CUDA 扩展
lib = load(
    name="matrix_transpose",
    # extra_include_paths=["include"],
    sources=["matrix_transpose.cu"],
    verbose=True
)


# 定义矩阵的维度
rows = 10
cols = 5

# 10*5
input = torch.full((rows, cols), 3.0, device='cuda')
print(input)

output= torch.empty((cols, rows), device='cuda')
lib.matrix_transpose(input, output, rows, cols)
# 5*10
print(output)



# Loading extension module matrix_transpose...
# tensor([[3., 3., 3., 3., 3.],
#         [3., 3., 3., 3., 3.],
#         [3., 3., 3., 3., 3.],
#         [3., 3., 3., 3., 3.],
#         [3., 3., 3., 3., 3.],
#         [3., 3., 3., 3., 3.],
#         [3., 3., 3., 3., 3.],
#         [3., 3., 3., 3., 3.],
#         [3., 3., 3., 3., 3.],
#         [3., 3., 3., 3., 3.]], device='cuda:0')
# tensor([[3., 3., 3., 3., 3., 3., 3., 3., 3., 3.],
#         [3., 3., 3., 3., 3., 3., 3., 3., 3., 3.],
#         [3., 3., 3., 3., 3., 3., 3., 3., 3., 3.],
#         [3., 3., 3., 3., 3., 3., 3., 3., 3., 3.],
#         [3., 3., 3., 3., 3., 3., 3., 3., 3., 3.]], device='cuda:0')


