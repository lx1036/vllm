

# https://leetgpu.com/challenges/relu

# Rectified Linear Unit (ReLU)


import torch
from torch.utils.cpp_extension import load

# JIT 编译并加载 CUDA 扩展
lib = load(
    name="relu",
    # extra_include_paths=["include"],
    sources=["relu.cu"],
    verbose=True
)


# N = 10
# A = torch.full(N, 3.0, device='cuda')

# 定义张量的长度
length = 10
# 创建一个值为-1和1相间隔的列表
values = [(-1) ** i for i in range(length)]
# 将列表转换为张量
input = torch.tensor(values, dtype=torch.float, device='cuda')
print(input)
output = torch.empty_like(input, dtype=torch.float, device='cuda')

lib.relu(input, output, length)
print(output)

# Loading extension module relu...
# tensor([ 1., -1.,  1., -1.,  1., -1.,  1., -1.,  1., -1.], device='cuda:0')
# tensor([1., 0., 1., 0., 1., 0., 1., 0., 1., 0.], device='cuda:0')

