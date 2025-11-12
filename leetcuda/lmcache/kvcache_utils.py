


import torch



def generate_tokens(num_tokens, device, fixed=False):
    if fixed:
        return torch.tensor([-1] * num_tokens).to(device)
    else:
        # random tokens
        return torch.randint(0, 10000, size=[num_tokens]).to(device)


def generate_kv_cache(num_tokens, fmt, device):
    # shape=model[32][2000][8][128], torch.bfloat16
    ret = []
    num_layers = 32
    num_heads = 8
    head_size = 128
    shape = ([num_tokens, num_heads, head_size] if fmt == "vllm" else [num_heads, num_tokens, head_size])
    dtype = torch.bfloat16 if fmt == "vllm" else torch.float16

    for i in range(num_layers):
        k = torch.rand(shape, dtype=dtype, device=device)
        v = torch.rand(shape, dtype=dtype, device=device)
        ret.append((k, v)) # ([layer_i][token_j][8][128], [layer_i][token_j][8][128])

    return tuple(ret)








