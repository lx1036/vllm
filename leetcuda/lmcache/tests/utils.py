


import torch

from leetcuda.lmcache.config import LMCacheEngineMetadata


def dumb_metadata(fmt="vllm", kv_shape=(32, 2, 256, 8, 128)):
    return LMCacheEngineMetadata("test_model", 3, 123, fmt, torch.bfloat16, kv_shape)

def generate_tokens(num_tokens, device, fixed=False):
    if fixed:
        return torch.tensor([-1] * num_tokens).to(device)
    else:
        # random tokens
        return torch.randint(0, 10000, size=[num_tokens]).to(device)


