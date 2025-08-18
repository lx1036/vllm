


#include <cuda_runtime.h>
#include <torch/extension.h>


__global__ void reverse_array2_kernel(float* input, int N) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;

  if(idx<N/2){
    float tmp = input[idx];
    input[idx] = input[N-1-idx];
    input[N-1-idx] = tmp;
  }
}

__global__ void reverse_array_kernel(float* input, float* output, int N) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;

  // N 是输入向量长度
  if (i >= N) {
    return;
  }

  output[i] = input[N - i - 1];
}


void reverse_array2(torch::Tensor input, int N) {
  // 定义线程块和网格大小
  const int threadsPerBlock = 256;
  const int blocksPerGrid = (N + threadsPerBlock - 1) / threadsPerBlock;

  reverse_array2_kernel<<<blocksPerGrid, threadsPerBlock>>>(
      input.data_ptr<float>(),
      N);
}

void reverse_array(torch::Tensor input, torch::Tensor output, int N) {
  // 定义线程块和网格大小
  const int threadsPerBlock = 256;
  const int blocksPerGrid = (N + threadsPerBlock - 1) / threadsPerBlock;

  reverse_array_kernel<<<blocksPerGrid, threadsPerBlock>>>(
      input.data_ptr<float>(),
      output.data_ptr<float>(),
      N);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("reverse_array", &reverse_array, "reverse array kernel (CUDA)");
  m.def("reverse_array2", &reverse_array2, "reverse array kernel (CUDA)");
}


