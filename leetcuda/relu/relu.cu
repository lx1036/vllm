


#include <cuda_runtime.h>
#include <torch/extension.h>

__global__ void relu_kernel(const float* input, float* output, int N) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;

  // N 是输入向量长度
  if (i >= N) {
    return;
  }

  if (input[i] < 0) {
    output[i] = 0;
  } else {
    output[i] = input[i];
  }
}


void relu(torch::Tensor A,torch::Tensor B, int N) {
  // 定义线程块和网格大小
//  dim3 threadsPerBlock(16, 16); // 二维线程块  16*16
  const int threadsPerBlock = 256;
  const int blocksPerGrid = (N + threadsPerBlock - 1) / threadsPerBlock;

  relu_kernel<<<blocksPerGrid, threadsPerBlock>>>(
      A.data_ptr<float>(),
      B.data_ptr<float>(),
      N);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("relu", &relu, "RELU(Rectified Linear Unit) kernel (CUDA)");
}


