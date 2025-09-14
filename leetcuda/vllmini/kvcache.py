from typing import Dict, List, Tuple

import torch


class KVCache:

    def __init__(self, num_blocks: int, num_heads: int, head_size: int, block_size: int, max_blocks_per_seq: int):
        self.block_size = block_size
        self.max_blocks_per_seq = max_blocks_per_seq
        self.free_blocks = list(range(num_blocks)) # [0, ..., num_blocks-1], []int
        self.allocated_blocks: Dict[int, List[int]] = {} # map[int][]int
        self.block_tables: Dict[int, List[Tuple[int, int]]] = {} # map[int][][2]int
        self.paged_attention_block_tables: Dict[int, List[List[int]]] = {} # map[int][][]int

        # key_cache = [num_blocks, num_heads, head_size // x, block_size, x] 这里把 head_size 也分成 x 组
        self.key_cache = torch.zeros(num_blocks, num_heads, head_size // 8, block_size, 8, dtype=torch.float16, device="cuda")
        self.value_cache = torch.zeros(num_blocks, num_heads, head_size, block_size, dtype=torch.float16, device="cuda")


    def allocate_for_prefill(self, seq_id: int, num_layers: int, seq_len: int) -> Tuple[List[int], List[torch.Tensor], List[List[int]]]:

        # 每一层分配一个 blocks=[id_1, id_2, ..., id_n]
        allocated = self.free_blocks[:num_layers]
        self.free_blocks = self.free_blocks[num_layers:]
        self.allocated_blocks[seq_id] = allocated

        # 这里 min(seq_len, self.block_size)，也是减少内存，可能 seq_len 都不到 self.block_size
        # 每一层都需要 prefill(input_tokens)
        self.block_tables[seq_id] = [(block, min(seq_len, self.block_size)) for block in allocated]
        self.paged_attention_block_tables[seq_id] = [
            torch.tensor([[block] + [-1]*(self.max_blocks_per_seq-1)], dtype=torch.int32, device="cuda") for block in allocated
        ]

        # 加上 block_size * block, 每个 block 里的 index 为 [0,1,2,..., seq_len-1]
        # []torch.Tensor
        slot_mappings = [torch.arange(seq_len, dtype=torch.long, device="cuda") + block * self.block_size for block in allocated]
        return allocated, slot_mappings, self.paged_attention_block_tables[seq_id]


    def free(self, seq_id: int):
        if seq_id in self.allocated_blocks:
            self.free_blocks.extend(self.allocated_blocks[seq_id])
            del self.allocated_blocks[seq_id]
            del self.block_tables[seq_id]
            del self.paged_attention_block_tables[seq_id]




