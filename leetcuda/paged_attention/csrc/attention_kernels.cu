

/**
 * 参考文献:
 * 源代码：https://github.com/vllm-project/vllm/blob/submission/csrc/attention_kernels.cu
 * 《vLLM Paged Attention代码分析》：https://mp.weixin.qq.com/s/UvJvn54ta9SFXyQV4ebjvg
 * 《小米二面问我PagedAttention》：https://mp.weixin.qq.com/s/2wEzZST8GAjx4nRebkLaRQ
 * 《PagedAttention官网解析》：https://docs.vllm.ai/en/stable/design/paged_attention.html
 * 《PageAttention V1 核心CUDA源代码阅读》：https://zhuanlan.zhihu.com/p/667417423
 * 《PagedAttention CUDA Kernel》：https://zhuanlan.zhihu.com/p/3179105297
 * 《PagedAttention CUDA Kernel-B站视频版》：https://www.bilibili.com/video/BV1Uq1cYiEPg/
 */



//#include <cuda_runtime.h>
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>


/**
 * A100:
 * (1)有108个SM(Streaming MultiProcessor)
 * (2)每个SM有4个Warp Scheduler，Threads以32个为一组同时执行TreadWrap, 所以同时有最多4个ThreadWrap同时在一个SM上执行。即一个SM总共128=32(1ThreadWrap)*4(warp scheduler)
 */

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
    const scalar_t* __restrict__ query, // [num_seqs(序列长度), num_heads(head数量), head_size(head维度)]
    const scalar_t* __restrict__ key_cache, // [num_blocks, num_heads, head_size/x, block_size(=16比如一个block有16个token), x], x优化了key_cache读取效率???
    const scalar_t* __restrict__ value_cache, // [num_blocks, num_heads, head_size, block_size(=16比如一个block有16个token)],
    const float scale,
    const int* __restrict__ block_tables, // [num_seqs, max_num_blocks_per_seq],
    const int* __restrict__ context_lens, // [num_seqs],
    const int max_num_blocks_per_seq,
    const int query_stride //
) {

    const int thread_idx = threadIdx.x;
    const int warp_idx = thread_idx / WARP_SIZE;
    const int lane = thread_idx % WARP_SIZE;
    const int head_idx = blockIdx.x;
    const int seq_idx = blockIdx.y; // ???
    const int num_heads = gridDim.x;



    // (1)THREAD_GROUP_SIZE 重要！！！这里以一个SM为视角：总共128个Thread，一个SM可以同时执行4组ThreadWrap,每组ThreadWrap有32个Threads，16个Token每个有2个Thread同时执行。
    constexpr int THREAD_GROUP_SIZE = MAX(WARP_SIZE / BLOCK_SIZE, 1); // 32/16==2，2组 *16token, WARP_SIZE=32是把32个Threads又分为一组同时执行
    constexpr int NUM_TOKENS_PER_THREAD_GROUP = (BLOCK_SIZE + WARP_SIZE - 1) / WARP_SIZE; // ==1
    constexpr int NUM_WARPS = NUM_THREADS / WARP_SIZE; // 4==128/32=4，正好 4 个 warp scheduler

    // ???
    constexpr int VEC_SIZE = MAX(16 / (THREAD_GROUP_SIZE * sizeof(scalar_t)), 1); //一个token有THREAD_GROUP_SIZE=2个读，总共一次性读16B，16/(2*8)


    // ???
    const int seq_idx = blockIdx.y;

    // 并行运行，找到 block_tables 里的 第 i 个 seq 对应的 block_table
    const int* block_table = block_tables + seq_idx * max_num_blocks_per_seq;

    // 这里是把 thread 又分组了???
    const int thread_group_idx = thread_idx / THREAD_GROUP_SIZE;

    /**
     * (1)读 Q 和 K
     */
     const int context_len = context_lens[seq_idx];
     const int num_blocks = (context_len + BLOCK_SIZE - 1) / BLOCK_SIZE;
    // 4个wrap一起遍历所有所有paged blocks, 每个wrap处理一个paged block, 4 个wrap同时执行 （并行度）
    for (int block_idx = warp_idx; block_idx < num_blocks; block_idx += NUM_WARPS) {
        const int physical_block_number = block_table[block_idx];

        for (int i = 0; i < NUM_TOKENS_PER_THREAD_GROUP; i++) {
            const int physical_block_offset = (thread_group_idx + i * WARP_SIZE) % BLOCK_SIZE;

        }

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
    int num_seqs = query.size(0); // query: [num_seqs(序列长度), num_heads(head数量), head_size(head维度)]
    int num_heads = query.size(1);
    int head_size = query.size(2);
    int max_num_blocks_per_seq = block_tables.size(1); //???
    int query_stride = query.stride(0); // 返回0维度的步长，即有多少维度



    int shared_mem_size = std::max(logits_size, outputs_size);

    // num_heads 表示处理的头数（heads）, num_seqs 表示处理的序列数（sequences）, 意味着整个网格由 num_heads * num_seqs 个线程块 block 组成
    dim3 grid(num_heads, num_seqs); // num_heads=8
    dim3 block(NUM_THREADS); // 一个SM总共128个Thread->4组ThreadWrap->每个token有2个thread
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

#define CALL_KERNEL_LAUNCHER(T, BLOCK_SIZE) \
single_query_cached_kv_attention_launcher<T, BLOCK_SIZE>(out, query, key_cache, value_cache, scale, block_tables, context_lens, max_context_len);


void single_query_cached_kv_attention(
        torch::Tensor& out,
        torch::Tensor& query,
        torch::Tensor& key_cache,
        torch::Tensor& value_cache,
        float scale,
        torch::Tensor& block_tables,
        torch::Tensor& context_lens,
        int block_size,
        int max_context_len) {
    if (query.element_size() == 2) {
        if (block_size == 1) {
            CALL_KERNEL_LAUNCHER(uint16_t, 1);
        } else if (block_size == 2) {
            CALL_KERNEL_LAUNCHER(uint16_t, 2);
        }



    } else {
        assert(false);
    }
}


PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {

}
