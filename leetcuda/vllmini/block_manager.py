from typing import Tuple, List

from .kvcache import KVCache

import torch


class BlockManager:
    def __init__(self, num_blocks: int, block_size: int, num_heads: int, head_size: int, max_blocks_per_seq: int):

        self.block_size = block_size
        self.num_heads = num_heads

        self.kv_cache: KVCache = KVCache(
            num_blocks=num_blocks,
            num_heads=num_heads,
            head_size=head_size,
            block_size=block_size,
            max_blocks_per_seq=max_blocks_per_seq,
        )




    def allocate_for_prefill(self, seq_id: int, num_layers: int, seq_len: int) -> Tuple[int, List[int], List[torch.Tensor], List[List[int]]]:
        allocated, slot_mappings, paged_attention_block_table = self.kv_cache.allocate_for_prefill(seq_id, num_layers, seq_len)
        return seq_id, allocated, slot_mappings, paged_attention_block_table


