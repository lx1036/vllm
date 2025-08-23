



import torch
from torch.utils.cpp_extension import load

# JIT 编译并加载 CUDA 扩展
lib = load(
    name="convolution_1d",
    # extra_include_paths=["include"],
    sources=["convolution_1d.cu"],
    verbose=True
)

input = torch.tensor([1,2,3,4,5], dtype=torch.float, device='cuda')
kernel = torch.tensor([1,0,-1], dtype=torch.float, device='cuda')
output = torch.tensor([0,0,0], dtype=torch.float, device='cuda')
print(len(input), len(kernel))

lib.convolution_1d(input, kernel, output, len(input), len(kernel))
print(output)


# Loading extension module convolution_1d...
# 5 3
# tensor([-2., -2., -2.], device='cuda:0')

