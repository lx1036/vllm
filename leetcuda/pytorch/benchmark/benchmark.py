


import torch
import time
import numpy as np

def measure_p2p_bandwidth():
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
                    tensors[j].copy_(tensors[i], non_blocking=True)
    torch.cuda.synchronize()

    bandwidth = np.zeros((num_gpus, num_gpus))
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
            bandwidth[i, j] = bw

    print("\n=== Summary ===")
    print("\nP2P bandwidth matrix (GB/s):")
    print("    " + " ".join([f"GPU {i:2d}" for i in range(num_gpus)]))
    for i in range(num_gpus):
        row = [f"{bandwidth[i, j]:6.2f}" for j in range(num_gpus)]
        print(f"GPU {i}: " + " ".join(row))

def measure_p2p_bandwidth2():
    dtype = torch.float32
    size = 1024*1024*128

    tensors = []
    iterations = 100
    num_gpus = torch.cuda.device_count()

    for i in range(num_gpus):
        # with torch.cuda.device(i):
        tensor = torch.randn(size, dtype=dtype, device=f'cuda:{i}')
        tensors.append(tensor)

    # # Warmup 后，速度更快!!!
    # warmup = 5
    # for _ in range(warmup):
    #     for i in range(num_gpus):
    #         for j in range(num_gpus):
    #             if i != j:
    #                 tensors[j].copy_(tensors[i], non_blocking=True)
    # torch.cuda.synchronize()


    start_time = time.time()
    for _ in range(iterations):
        start_time1 = time.time()
        tensors[2].copy_(tensors[3], non_blocking=False) # i->j
        end_time1 = time.time()
        total_time1 = end_time1 - start_time1
        print(f"total_time: {total_time1}")

    torch.cuda.synchronize()

    end_time = time.time()
    total_time = end_time - start_time
    print(f"sum total_time: {total_time}")

    element_size = torch.tensor([], dtype=dtype).element_size()
    data_size_bytes = size * element_size
    data_size_gb = data_size_bytes / (1024**3) # 0.5 GB
    total_data = data_size_gb * iterations
    print(f"total_data: {total_data} GB") # 10GB

    bw = total_data / total_time  # GB/s
    print(f"GPU {3} -> GPU {2}: {bw:.2f} GB/s") # 1370.78 GB/s




def memory_bandwidth_benchmark():
    test_range = 2 ** (np.arange(20, 27, 0.5))
    num_trails = 10

    print('size (GB), elapsed_time, bandwidth (GB/s)')
    for size in test_range:
        elapsed_time = 0
        for _ in range(num_trails):
            size = int(size)

            # Create random tensors
            a = torch.rand(size, device="cuda:1")
            b = torch.rand(size, device="cuda:2")

            # Warm-up to ensure CUDA kernel is initialized if using GPU
            torch.cuda.synchronize()
            a.copy_(b)
            torch.cuda.synchronize()

            # Record the start time
            start_time = time.time()

            # Perform the copy operation
            a.copy_(b)

            # Synchronize if using CUDA to make sure operation is finished
            torch.cuda.synchronize()

            # Record the end time
            end_time = time.time()

            # Compute elapsed time
            elapsed_time += end_time - start_time

        elapsed_time = elapsed_time / num_trails
        # Calculate Bandwidth in GB/s
        bytes_copied = a.nelement() * a.element_size()  # bytes
        bandwidth = 2 * bytes_copied / elapsed_time / 1e9  # GB/s

        print(bytes_copied / 1e9, elapsed_time, bandwidth, sep=', ')


if __name__ == "__main__":
    # measure_p2p_bandwidth()
    # measure_p2p_bandwidth2()
    memory_bandwidth_benchmark()
