


#include <cuda_runtime.h>
#include <torch/extension.h>

__global__ void vector_add_kernel(const float* a, const float* b, float* out, int n) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) {
    out[i] = a[i] + b[i];
  }
}

// PyTorch 接口函数
void vector_add(torch::Tensor a, torch::Tensor b, torch::Tensor out) {
  int n = a.size(0);
  const int threads = 256;
  const int blocks = (n + threads - 1) / threads; // (1024+256-1)/256=4
  vector_add_kernel<<<blocks, threads>>>(
      a.data_ptr<float>(),
      b.data_ptr<float>(),
      out.data_ptr<float>(),
      n
  );
}


// https://leetgpu.com/challenges
__global__ void vector_add2(const float* A, const float* B, float* C, int N) {
  // Calculate global thread ID
  int idx = blockIdx.x * blockDim.x + threadIdx.x;

  if (idx < N) {
    C[idx] = A[idx] + B[idx];
  }
}

// A, B, C are device pointers (i.e. pointers to memory on the GPU)
void solve(const float* A, const float* B, float* C, int N) {
  int threadsPerBlock = 256;
  int blocksPerGrid = (N + threadsPerBlock - 1) / threadsPerBlock;

  vector_add2<<<blocksPerGrid, threadsPerBlock>>>(A, B, C, N);
  cudaDeviceSynchronize();
}


PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("vector_add", &vector_add, "Vector add kernel (CUDA)");
}
