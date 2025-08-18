

#include <cuda_runtime.h>
#include <torch/extension.h>

// 矩阵相乘
// C=A*B, A=M*N, B=N*K
__global__ void matrix_multiple_kernel(const float* A, const float* B, float* C, int M, int N, int K) {
  int row = blockIdx.y * blockDim.y + threadIdx.y;
  int col = blockIdx.x * blockDim.x + threadIdx.x;

  if (row < M && col < K) {
    float sum = 0;
    for(int i = 0; i < N; i++) {
      sum += A[row * N + i] * B[i * K + col];
    }

    C[row * K + col] = sum;
  }
}

void matrix_multiple(torch::Tensor A, torch::Tensor B, torch::Tensor C, int M, int N, int K) {
  // 定义线程块和网格大小
  dim3 threadsPerBlock(16, 16);
  dim3 numBlocks((K + threadsPerBlock.x - 1) / threadsPerBlock.x, (M + threadsPerBlock.y - 1) / threadsPerBlock.y);

//  dim3 block(16, 16);
//  dim3 grid((K + block.x - 1) / block.x, (M + block.y - 1) / block.y);

//  const int threads = 256;
//  const int blocks = (n + threads - 1) / threads; // (1024+256-1)/256=4

  matrix_multiple_kernel<<<numBlocks, threadsPerBlock>>>(
      A.data_ptr<float>(),
      B.data_ptr<float>(),
      C.data_ptr<float>(),
      M, N, K);
//  matrix_multiple_kernel<<<grid, block>>>(A, B, C, M, N, K);
}


PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("matrix_multiple", &matrix_multiple, "Matrix multiple kernel (CUDA)");
}
