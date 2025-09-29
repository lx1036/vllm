


import torch
import time
import numpy as np

def measure_p2p_bandwidth():
    dtype = torch.float32
    size = 1024*1024*16

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
                    tensors[j].copy_(tensors[i], non_blocking=False)
    torch.cuda.synchronize()

    bandwidth = np.zeros((num_gpus, num_gpus))
    for i in range(num_gpus-1, -1, -1):
        for j in range(num_gpus-1, -1, -1):
            if i == j:
                continue

            start_time = time.time()
            for _ in range(iterations):
                tensors[j].copy_(tensors[i], non_blocking=False) # i->j
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
    print("    " + " ".join([f"GPU{i:2d}" for i in range(num_gpus)]))
    for i in range(num_gpus):
        row = [f"{bandwidth[i, j]:6.2f}" for j in range(num_gpus)]
        print(f"GPU{i}: " + " ".join(row))



if __name__ == "__main__":
    measure_p2p_bandwidth()
