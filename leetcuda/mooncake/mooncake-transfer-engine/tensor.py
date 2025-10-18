


import torch
from safetensors.torch import load as safetensors_load
from safetensors.torch import save as safetensors_save

server_buffer = torch.ones(10, device="cpu")
server_ptr = server_buffer.data_ptr()
server_len = server_buffer.size().numel()
print(server_ptr, server_len)
user_data = safetensors_save({"tensor": server_buffer})
print(user_data, len(user_data))

print(safetensors_load(user_data)["tensor"])
'''
tensor([1., 1., 1., 1., 1., 1., 1., 1., 1., 1.])
'''
