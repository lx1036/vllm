

// 参考文献:
// https://github.com/vllm-project/vllm/blob/submission/csrc/attention_kernels.cu

//#include <cuda_runtime.h>
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>




// GPU 硬件基础知识：https://zhuanlan.zhihu.com/p/123170285
// SM采用SIMT(Single-Instruction-Multiple-Thread,单指令多线程)架构，warp(=32*threads)线程束是最基本执行单元，**这些treads以不同数据源执行相同指令**。
// 一个 SM(Streaming MultiProcessor) 只有4个 warp scheduler。
// 一个 SM(Streaming MultiProcessor) 由多个 CUDA core 组成，比如 Pascal 架构 GPU 有 128 个 CUDA Core。
// SM 还包括特殊运算单元(SFU)，共享内存(shared memory)，寄存器文件(Register File)和调度器(Warp Scheduler)等等
#define WARP_SIZE 32


namespace PagedAttention {

/**
 * 模板允许你编写泛型代码，这些代码可以用于多种数据类型,你可以在编译时生成针对不同数据类型的代码，而不需要为每种类型单独编写代码。
 */
template<
    typename scalar_t,
    int HEAD_SIZE,
    int BLOCK_SIZE,
    int NUM_THREADS>
__global__ void single_query_cached_kv_attention_kernel(
    // 表示 out 指针是唯一访问其指向的内存区域的指针
    // 特别是在并行计算和高性能计算中，使用 __restrict__ 可以显著提高代码的执行效率
    scalar_t* __restrict__ output, // [num_seqs(序列长度), num_heads=8, head_size=64, 所以总维度是512=8*64],
    const scalar_t* __restrict__ query, // [num_seqs(序列长度), num_heads, head_size],
    const scalar_t* __restrict__ key_cache, // [num_blocks, num_heads, head_size/x, block_size(=16比如一个block有16个token), x],
    const scalar_t* __restrict__ value_cache, // [num_blocks, num_heads, head_size, block_size(=16比如一个block有16个token)],
    const float scale,
    const int* __restrict__ block_tables, // [num_seqs, max_num_blocks_per_seq],
    const int* __restrict__ context_lens, // [num_seqs],
    const int max_num_blocks_per_seq,
    const int query_stride
) {


    // 这里block_size一般是4/8/16这样子,比如一个block有16个token。表示有多少个 warp 可以并行执行，但是只能被 4 个 warp scheduler 调度。
    // BLOCK_SIZE=16 tokens per warp
    constexpr int THREAD_GROUP_SIZE = MAX(WARP_SIZE / BLOCK_SIZE, 1); // ==2，2组 *16token
    constexpr int NUM_TOKENS_PER_THREAD_GROUP = (BLOCK_SIZE + WARP_SIZE - 1) / WARP_SIZE; // ==1
    constexpr int NUM_WARPS = NUM_THREADS / WARP_SIZE; // 4==128/32=4，正好 4 个 warp scheduler

    // ???
    const int seq_idx = blockIdx.y;

    // 并行运行，找到 block_tables 里的 第 i 个 seq 对应的 block_table
    const int* block_table = block_tables + seq_idx * max_num_blocks_per_seq;

    /**
     * (1)读 Q 和 K
     */
    // 4个wrap一起遍历所有所有paged blocks, 每个wrap处理一个paged block, 4 个wrap同时执行 （并行度）
    for (int block_idx = warp_idx; block_idx < num_blocks; block_idx += NUM_WARPS) {
        const int physical_block_number = block_table[block_idx];

    }



}

} // namespace PagedAttention



void single_query_cached_kv_attention_kernel();


#define LAUNCH_ATTENTION_KERNEL(T, HEAD_SIZE, BLOCK_SIZE, NUM_THREADS) PagedAttention::single_query_cached_kv_attention_kernel<T, HEAD_SIZE, BLOCK_SIZE, NUM_THREADS> \
    <<<grid, block, shared_mem_size, stream>>>(out_ptr, query_ptr, key_cache_ptr, value_cache_ptr, scale, block_tables_ptr, context_lens_ptr, max_num_blocks_per_seq, query_stride);

template<
    typename T,
    int BLOCK_SIZE,
    int NUM_THREADS = 128 // 128 threads per block
    >
void single_query_cached_kv_attention_launcher(
    torch::Tensor& out,
    torch::Tensor& query,
    torch::Tensor& key_cache,
    torch::Tensor& value_cache,
    float scale,
    torch::Tensor& block_tables,
    torch::Tensor& context_lens,
    int max_context_len) {



    int shared_mem_size = std::max(logits_size, outputs_size);

    // num_heads 表示处理的头数（heads）, num_seqs 表示处理的序列数（sequences）, 意味着整个网格由 num_heads * num_seqs 个线程块 block 组成
    dim3 grid(num_heads, num_seqs); // num_heads=8
    dim3 block(NUM_THREADS);
    /**
     * dim3 grid(num_heads, num_seqs);  // 定义网格，8个头，10个序列
       dim3 block(NUM_THREADS);         // 定义线程块，每个块128个线程
       调用内核函数,内核函数会被启动 80 次（每个线程块一次），每次启动时，每个线程块包含 128 个线程。
        single_query_cached_kv_attention_kernel<<<grid, block>>>(out, q, k_cache, v_cache);
     */
    const cudaStream_t stream = at::cuda::getCurrentCUDAStream(); // ???
    switch (head_size) {
      case 32:
        LAUNCH_ATTENTION_KERNEL(T, 32, BLOCK_SIZE, NUM_THREADS);
        break;

      case 64: // 512/8(heads)=64(head_size)
        LAUNCH_ATTENTION_KERNEL(T, 64, BLOCK_SIZE, NUM_THREADS);
        break;

      default:
        assert(false);
        break;
    }
}




PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {

}
