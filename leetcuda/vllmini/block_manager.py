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

    def decode_step(self, seq_id: int, input_len: int):


        block_table = self.kv_cache.get_block_table(seq_id)
        paged_attention_block_table = self.kv_cache.get_paged_attention_block_table(seq_id)


        for block_id, layer_blocks in enumerate(paged_attention_block_table):
            last_block = -1
            for i in range(1, len(layer_blocks[0])):
                if layer_blocks[0][i] == -1:
                    last_block = layer_blocks[0][i-1]
                    break















