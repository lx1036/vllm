

# import numpy as np
#
# test_range = 2 ** (np.arange(8, 13, 0.25)) # 生成一个 8 到 13 的等差数列，步长为 0.25
# print(test_range)
# for n in test_range:
#     print(n)


'''
这个基准测试代码包含以下主要功能：

1、测量单个 GPU 的内存带宽 - 通过在同一设备上复制张量来评估设备内部的内存性能
2、测量 GPU 之间的 P2P（点对点）带宽 - 评估不同 GPU 之间直接数据传输的速度
3、支持不同的数据类型和张量大小，以便进行全面测试


# 默认参数运行
python gpu_bandwidth_benchmark.py

# 自定义参数运行
python gpu_bandwidth_benchmark.py --size 268435456 --dtype float16 --iterations 30
'''


import torch
import time
import argparse
import numpy as np

def parse_args():
    parser = argparse.ArgumentParser(description='PyTorch Multi-GPU Memory Bandwidth Benchmark')
    parser.add_argument('--size', type=int, default=1024*1024*128,
                        help='Size of tensor in elements (default: 134,217,728)')
    parser.add_argument('--warmup', type=int, default=5,
                        help='Number of warmup iterations')
    parser.add_argument('--iterations', type=int, default=20,
                        help='Number of benchmark iterations')
    parser.add_argument('--dtype', type=str, default='float32',
                        help='Data type for tensors (float32, float16, etc.)')
    return parser.parse_args()

def get_dtype(dtype_str):
    """Convert string to PyTorch dtype"""
    if dtype_str == 'float32':
        return torch.float32
    elif dtype_str == 'float16':
        return torch.float16
    elif dtype_str == 'float64':
        return torch.float64
    elif dtype_str == 'int32':
        return torch.int32
    elif dtype_str == 'int64':
        return torch.int64
    else:
        raise ValueError(f"Unsupported dtype: {dtype_str}")

def measure_p2p_bandwidth(size, dtype, warmup=5, iterations=20):
    """Measure peer-to-peer bandwidth between all pairs of GPUs"""
    num_gpus = torch.cuda.device_count()
    if num_gpus < 2:
        print("Need at least 2 GPUs for P2P bandwidth measurement")
        return None

    dtype = get_dtype(dtype)
    element_size = torch.tensor([], dtype=dtype).element_size()
    data_size_bytes = size * element_size
    data_size_gb = data_size_bytes / (1024**3)

    # Initialize tensors on each GPU
    tensors = []
    for i in range(num_gpus):
        with torch.cuda.device(i):
            tensor = torch.randn(size, dtype=dtype, device='cuda')
            tensors.append(tensor)

    # Warmup
    for _ in range(warmup):
        for i in range(num_gpus):
            for j in range(num_gpus):
                if i != j:
                    tensors[i].copy_(tensors[j], non_blocking=True)
    torch.cuda.synchronize()

    # Benchmark
    bandwidth = np.zeros((num_gpus, num_gpus))

    for i in range(num_gpus):
        for j in range(num_gpus):
            if i == j:
                continue

            start_time = time.time()
            for _ in range(iterations):
                tensors[i].copy_(tensors[j], non_blocking=False)
            torch.cuda.synchronize()
            end_time = time.time()

            total_time = end_time - start_time
            total_data = data_size_gb * iterations
            bw = total_data / total_time  # GB/s

            bandwidth[i, j] = bw
            print(f"GPU {i} -> GPU {j}: {bw:.2f} GB/s")

    return bandwidth



def measure_p2p_bandwidth2():
    dtype = torch.float32
    size = 1024*1024*128

    # num_gpus = 8
    num_gpus = torch.cuda.device_count()
    tensors = []
    iterations = 20
    for i in range(num_gpus):
        with torch.cuda.device(i):
            tensor = torch.randn(size, dtype=dtype, device='cuda')
            tensors.append(tensor)

    # Warmup 后，速度更快!!!
    warmup = 5
    for _ in range(warmup):
        for i in range(num_gpus):
            for j in range(num_gpus):
                if i != j:
                    tensors[i].copy_(tensors[j], non_blocking=True)
    torch.cuda.synchronize()

    for i in range(num_gpus):
        for j in range(num_gpus):
            if i == j:
                continue

            start_time = time.time()
            for _ in range(iterations):
                tensors[j].copy_(tensors[i], non_blocking=True) # i->j
            torch.cuda.synchronize()

            end_time = time.time()
            total_time = end_time - start_time

            element_size = torch.tensor([], dtype=dtype).element_size()
            data_size_bytes = size * element_size
            data_size_gb = data_size_bytes / (1024**3) # 0.25 GB
            total_data = data_size_gb * iterations

            bw = total_data / total_time  # GB/s
            print(f"GPU {i} -> GPU {j}: {bw:.2f} GB/s") # 1370.78 GB/s



def measure_device_bandwidth(size, dtype, warmup=5, iterations=20):
    """Measure memory bandwidth on individual devices"""
    num_gpus = torch.cuda.device_count()
    dtype = get_dtype(dtype)
    element_size = torch.tensor([], dtype=dtype).element_size()
    data_size_bytes = size * element_size
    data_size_gb = data_size_bytes / (1024**3)

    bandwidths = []

    for i in range(num_gpus):
        with torch.cuda.device(i):
            # Create source and destination tensors
            src = torch.randn(size, dtype=dtype, device='cuda')
            dst = torch.empty_like(src)

            # Warmup
            for _ in range(warmup):
                dst.copy_(src, non_blocking=True)
            torch.cuda.synchronize()

            # Benchmark
            start_time = time.time()
            for _ in range(iterations):
                dst.copy_(src, non_blocking=True)
            torch.cuda.synchronize()
            end_time = time.time()

            total_time = end_time - start_time
            total_data = data_size_gb * iterations
            bw = total_data / total_time  # GB/s

            bandwidths.append(bw)
            print(f"GPU {i} memory bandwidth: {bw:.2f} GB/s")

    return bandwidths

def main():
    args = parse_args()

    # Check CUDA availability
    if not torch.cuda.is_available():
        print("CUDA is not available. Exiting.")
        return

    num_gpus = torch.cuda.device_count()
    print(f"Found {num_gpus} CUDA device(s)")
    for i in range(num_gpus):
        print(f"GPU {i}: {torch.cuda.get_device_name(i)}")

    dtype = get_dtype(args.dtype)
    element_size = torch.tensor([], dtype=dtype).element_size()
    data_size_bytes = args.size * element_size
    print(f"\nBenchmarking with tensor size: {args.size:,} elements ({data_size_bytes/1024**2:.2f} MB) of type {args.dtype}")

    print("\n=== Measuring device memory bandwidth ===")
    device_bw = measure_device_bandwidth(args.size, args.dtype, args.warmup, args.iterations)


    if num_gpus >= 2:
        print("\n=== Measuring P2P memory bandwidth ===")
        p2p_bw = measure_p2p_bandwidth(args.size, args.dtype, args.warmup, args.iterations)

        print("\n=== Summary ===")
        print("Device memory bandwidth:")
        for i, bw in enumerate(device_bw):
            print(f"GPU {i}: {bw:.2f} GB/s")

        print("\nP2P bandwidth matrix (GB/s):")
        print("    " + "   ".join([f"GPU{i:2d}" for i in range(num_gpus)]))
        for i in range(num_gpus):
            row = [f"{p2p_bw[i, j]:6.2f}" for j in range(num_gpus)]
            print(f"GPU {i}: " + " ".join(row))

if __name__ == "__main__":
    measure_p2p_bandwidth2()
    # main()
