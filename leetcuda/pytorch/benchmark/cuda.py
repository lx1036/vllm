



import torch
# print(torch.cuda.is_available())

dtype = torch.float16
size = 1024*1024*128

element_size = torch.tensor([], dtype=dtype).element_size()
data_size_bytes = size * element_size
data_size_gb = data_size_bytes / (1024**3)
print(data_size_gb) # 0.25
