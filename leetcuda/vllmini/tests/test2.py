import time
from typing import Dict, List

import torch


print(int(time.time() * 1000000))
print(time.time())
'''
1757773405636009
1757773405.6360288
'''


device = "cuda" if torch.cuda.is_available() else "cpu"

seq_len = 3
position_ids = torch.arange(seq_len, device=device)
print(position_ids)
'''
tensor([0, 1, 2], device='cuda:0')
'''

allocated = [10,11,12]
block_size = 4
slot_mappings = [torch.arange(seq_len, dtype=torch.long, device=device) + block * block_size for block in allocated] # 40 44 48
print(slot_mappings)
'''
[tensor([40, 41, 42], device='cuda:0'), tensor([44, 45, 46], device='cuda:0'), tensor([48, 49, 50], device='cuda:0')]
'''

num_blocks = 30
free_blocks = list(range(num_blocks))
print(free_blocks)
'''
[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29]
'''

seq_id = 3
max_blocks_per_seq = 10
paged_attention_block_tables: Dict[int, List[List[int]]] = {} # map[int][][]int
paged_attention_block_tables[seq_id] = [
    torch.tensor([[block] + [-1]*(max_blocks_per_seq-1)], dtype=torch.int32, device=device) for block in allocated
]
print(paged_attention_block_tables)
'''
{3: [
tensor([[10, -1, -1, -1, -1, -1, -1, -1, -1, -1]], device='cuda:0',dtype=torch.int32), 
tensor([[11, -1, -1, -1, -1, -1, -1, -1, -1, -1]], device='cuda:0',dtype=torch.int32), 
tensor([[12, -1, -1, -1, -1, -1, -1, -1, -1, -1]], device='cuda:0',dtype=torch.int32),
]}
'''

for block_id, block in enumerate(paged_attention_block_tables[seq_id]):
    print(block_id, block, block[0])
'''
0 tensor([[10, -1, -1, -1, -1, -1, -1, -1, -1, -1]], dtype=torch.int32) tensor([10, -1, -1, -1, -1, -1, -1, -1, -1, -1], dtype=torch.int32)
1 tensor([[11, -1, -1, -1, -1, -1, -1, -1, -1, -1]], dtype=torch.int32) tensor([11, -1, -1, -1, -1, -1, -1, -1, -1, -1], dtype=torch.int32)
2 tensor([[12, -1, -1, -1, -1, -1, -1, -1, -1, -1]], dtype=torch.int32) tensor([12, -1, -1, -1, -1, -1, -1, -1, -1, -1], dtype=torch.int32)
'''

logits = torch.ones([1,2,3]) # (1,2,3)
print(logits)
logits = logits[:, -1, :] # (1,3)
print(logits)

seq_lens = torch.tensor([seq_len], dtype=torch.int32, device=device)
print(seq_lens)
'''
tensor([3], device='cuda:0', dtype=torch.int32)
'''
