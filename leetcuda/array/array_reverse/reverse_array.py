

# https://leetgpu.com/challenges/reverse-array

import torch
from torch.utils.cpp_extension import load

# JIT 编译并加载 CUDA 扩展
lib = load(
    name="reverse_array",
    # extra_include_paths=["include"],
    sources=["reverse_array.cu"],
    verbose=True
)


# 定义张量的长度
length = 10
# 生成一个从0到length-1的张量
input = torch.arange(length, dtype=torch.float, device='cuda')
output = torch.empty_like(input, dtype=torch.float, device='cuda')

lib.reverse_array(input, output, length)
print(input, output)

lib.reverse_array2(input, length)
print(input)



# Loading extension module reverse_array...
# tensor([0., 1., 2., 3., 4., 5., 6., 7., 8., 9.], device='cuda:0') tensor([9., 8., 7., 6., 5., 4., 3., 2., 1., 0.], device='cuda:0')
# tensor([9., 8., 7., 6., 5., 4., 3., 2., 1., 0.], device='cuda:0')

