


#include <cuda_runtime.h>
#include <torch/extension.h>

// https://leetgpu.com/challenges/1d-convolution

// output[i]=Sum[j=0, kernel_size-1]input[i+j]*kernel[j]

// 一维卷积
__global__ void convolution_1d_kernel(const float* input, const float* kernel, float* output,
                                      int input_size, int kernel_size) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  float op = 0;
  if(idx < input_size - kernel_size + 1) {
    for(int j = 0; j < kernel_size; j++) {
      op += input[idx + j] * kernel[j];
    }
    output[idx] = op;
  }
}


void convolution_1d(torch::Tensor input, torch::Tensor kernel, torch::Tensor output, int input_size, int kernel_size) {
  const int threads = 256;
  int output_size = input_size - kernel_size + 1;
  const int blocks = (output_size + threads - 1) / threads; // (1024+256-1)/256=4
  convolution_1d_kernel<<<blocks, threads>>>(
      input.data_ptr<float>(),
      kernel.data_ptr<float>(),
      output.data_ptr<float>(),
      input_size, kernel_size
  );
}


PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("convolution_1d", &convolution_1d, "convolution_1d kernel (CUDA)");
}
