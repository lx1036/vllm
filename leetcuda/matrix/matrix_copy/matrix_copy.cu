

// https://leetgpu.com/challenges/matrix-copy


#include <cuda_runtime.h>
#include <torch/extension.h>



__global__ void matrix_copy_kernel(const float* A, float* B, int N) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < N*N) {
    B[idx] = A[idx];
  }
}


void matrix_copy(torch::Tensor input, torch::Tensor output, int N) {
  const int threads = 256;
  int total = N*N;
  const int blocks = (total + threads - 1) / threads; // (1024+256-1)/256=4
  matrix_copy_kernel<<<blocks, threads>>>(
      input.data_ptr<float>(),
      output.data_ptr<float>(),
      N
  );
}


PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("matrix_copy", &matrix_copy, "Vector add kernel (CUDA)");
}
