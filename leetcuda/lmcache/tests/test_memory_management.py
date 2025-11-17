from typing import Tuple

import torch

from leetcuda.lmcache.memory_management import TensorMemoryAllocator


def test_tensor_allocator():



    # 32 MB
    total_size = 1024 * 1024 * 32
    tensor_buffer = torch.zeros(total_size, dtype=torch.uint8, device="cpu")

    allocator = TensorMemoryAllocator(tensor_buffer)

    # 512 * 512 * 4 = 1MB
    print(f"torch.float.itemsize: {torch.float.itemsize}") # 4
    memory_obj1 = allocator.allocate([512, 512], dtype=torch.float)
    assert memory_obj1 is not None
    assert memory_obj1.tensor.dtype == torch.float
    assert memory_obj1.tensor.shape == (512, 512)





