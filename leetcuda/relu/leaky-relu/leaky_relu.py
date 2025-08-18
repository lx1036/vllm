




# https://leetgpu.com/challenges/leaky-relu

# Rectified Linear Unit (ReLU)


import torch
from torch.utils.cpp_extension import load

# JIT 编译并加载 CUDA 扩展
lib = load(
    name="leaky_relu",
    # extra_include_paths=["include"],
    sources=["leaky_relu.cu"],
    verbose=True
)

# 定义张量的长度
length = 10
# 创建一个值为-1和1相间隔的列表
values = [(-1) ** i for i in range(length)]
# 将列表转换为张量
input = torch.tensor(values, dtype=torch.float, device='cuda')
print(input)
output = torch.empty_like(input, dtype=torch.float, device='cuda')

lib.leaky_relu(input, output, length)
print(output)



# Loading extension module leaky_relu...
# tensor([ 1., -1.,  1., -1.,  1., -1.,  1., -1.,  1., -1.], device='cuda:0')
# tensor([ 1.0000, -0.0100,  1.0000, -0.0100,  1.0000, -0.0100,  1.0000, -0.0100,
#          1.0000, -0.0100], device='cuda:0')

