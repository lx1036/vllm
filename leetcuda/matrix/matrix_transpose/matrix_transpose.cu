



#include <cuda_runtime.h>
#include <torch/extension.h>



__global__ void matrix_transpose_kernel(const float* input, float* output, int rows, int cols) {
  int row = blockIdx.y * blockDim.y + threadIdx.y;
  int col = blockIdx.x * blockDim.x + threadIdx.x;
  if (row < rows && col < cols) {
    output[rows * col + row] = input[cols * row + col];
  }
}

void matrix_transpose(torch::Tensor input, torch::Tensor output, int rows, int cols) {
  // 定义线程块和网格大小
  dim3 threadsPerBlock(16, 16);
  dim3 blocksPerGrid((cols + threadsPerBlock.x - 1) / threadsPerBlock.x,
                     (rows + threadsPerBlock.y - 1) / threadsPerBlock.y);

  matrix_transpose_kernel<<<blocksPerGrid, threadsPerBlock>>>(
      input.data_ptr<float>(),
      output.data_ptr<float>(),
      rows, cols);
}


PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("matrix_transpose", &matrix_transpose, "Matrix Transpose kernel (CUDA)");
}

