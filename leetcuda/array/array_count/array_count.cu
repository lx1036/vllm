

// https://leetgpu.com/challenges/count-array-element


#include <cuda_runtime.h>
#include <torch/extension.h>

__global__ void array_count_kernel(const int* a, int* out, int N, int value) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < N) {
    if (a[i] == value) {
      atomicAdd(out, 1);
    }
  }
}

// PyTorch 接口函数
void array_count(torch::Tensor a, int* out, int N, int value) {
  const int threads = 256;
  const int blocks = (N + threads - 1) / threads; // (1024+256-1)/256=4
  array_count_kernel<<<blocks, threads>>>(
      a.data_ptr<int>(),
      out, N, value);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("array_count", &array_count, "Array count kernel (CUDA)");
}

