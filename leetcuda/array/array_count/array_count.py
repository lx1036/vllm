


import torch
from torch.utils.cpp_extension import load

# JIT 编译并加载 CUDA 扩展
lib = load(
    name="array_count",
    # extra_include_paths=["include"],
    sources=["array_count.cu"],
    verbose=True
)

# 定义张量的长度
length = 10
# 生成一个从0到length-1的张量
input = torch.arange(length, dtype=torch.int, device='cuda')
print(input)

count = 0
value = 9
lib.array_count(input, count, length, value)
print(count)
