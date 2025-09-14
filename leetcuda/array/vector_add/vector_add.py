
import torch
from torch.utils.cpp_extension import load

# JIT 编译并加载 CUDA 扩展
lib = load(
    name="vector_add",
    # extra_include_paths=["include"],
    sources=["vector_add.cu"],
    verbose=True
)

# 测试用法
a = torch.randn(8, device='cuda') # 1024
b = torch.randn(8, device='cuda')
out = torch.empty_like(a)

lib.vector_add(a, b, out)
print(out[:5], (a + b)[:5])  # 检查正确性


# Loading extension module vector_add...
# tensor([ 0.2047, -1.3533, -0.8366, -0.4598, -0.1486], device='cuda:0') tensor([ 0.2047, -1.3533, -0.8366, -0.4598, -0.1486], device='cuda:0')


# 编写PyTorch CUDA算子的关键在于：
# 1. 理解CUDA编程模型，编写高效的CUDA kernel。
# 2. 使用PyTorch的C++ API ( torch::extension ) 封装CUDA kernel，使其能够与PyTorch的张量交互。
# 3. 使用setup.py编译并生成Python可以调用的模块。
# 4. 在Python代码中导入并调用编译好的模块。

# 编译日志类似
# [1/2] nvcc -c add2_kernel.cu -o add2_kernel.cuda.o
# [2/3] c++ -c add2.cpp -o add2.o
# [3/3] c++ add2.o add2_kernel.cuda.o -shared -o add2.so


