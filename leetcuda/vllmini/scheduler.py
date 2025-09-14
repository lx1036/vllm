import time
from queue import PriorityQueue
from typing import Dict

import torch
import torch.nn.functional as F

from .gpt2 import GPT2LMHeadModel
from block_manager import BlockManager

class Scheduler:

    def __init__(self, model: GPT2LMHeadModel, block_manager: BlockManager, max_length: int): # max_length 为每个 seq 最大输出 token 数量
        self.model = model
        self.model = self.model.to(torch.float16) # 使用 float16 精度
        self.block_manager = block_manager
        self.max_length = max_length
        self.queue = PriorityQueue()
        self.active_seqs: Dict[int, float] = {} # {seq_id: arrival_time}

        self.last_logits: Dict[int, torch.Tensor] = {}
        self.seq_lens: Dict[int, int] = {} # {seq_id: length}
        self.seqs: Dict[int, torch.Tensor] = {} # {seq_id: tokens]



    # input_ids: [我 爱 吃 酸菜鱼]
    def add_sequence(self, input_ids: torch.Tensor):
        arrival_time = time.time()
        seq_id = self.generate_seq_id()
        self.queue.put((arrival_time, seq_id))
        self.active_seqs[seq_id] = arrival_time

        # [batch_size, seq_len, hidden_size]
        seq_len = input_ids.size(1)
        num_layers = len(self.model.transformer.h)

        seq_id, _, slot_mappings, paged_attention_block_table = self.block_manager.allocate_for_prefill(seq_id, num_layers, seq_len)
        # key_cache, value_cache 都为空值
        key_cache, value_cache = self.block_manager.kv_cache.key_cache, self.block_manager.kv_cache.value_cache
        # (batch_size, num_heads, seq_len, seq_len)
        attention_mask = generate_triangular_mask(1, self.block_manager.num_heads, seq_len)

        seq_len = input_ids.size(1) # [batch_size, seq_len]
        logits = self.model(
            input_ids=input_ids,
            position_ids=torch.arange(seq_len, device=input_ids.device), # 生成一个初始化的 position_ids tensor，后续会从模型权重里加载对应的真实的值
            attention_mask=attention_mask,
            use_cache=True,
            is_prefill=True,
            key_cache=key_cache,
            value_cache=value_cache,
            slot_mappings=slot_mappings,
            block_tables=paged_attention_block_table,
        )
        self.last_logits[seq_id] = logits[:, -1, :]
        # 记录该 seq 的 tokens 个数
        self.seq_lens[seq_id] = seq_len
        self.seqs[seq_id] = input_ids
        print(f"Prefill seq {seq_id} len {seq_len} is complete successfully")
        return seq_id


    def run(self):
        while not self.queue.empty():
            print(f"remaining seqs number need be processed: {self.queue.qsize()}")
            _, seq_id = self.queue.get()
            if seq_id not in self.active_seqs:
                print(f"seq_id {seq_id} is not in active_seqs, skip it")
                continue

            try:

                next_token = self.sample_next_token(seq_id)
                # TODO: decode from next_token
                current_tokens = self.seqs[seq_id]
                input_ids = next_token.unsqueeze(0)
                # 将当前序列 current_tokens 和新生成的 next_token 拼接在一起，形成新的的 tokens
                self.seqs[seq_id] = torch.cat([current_tokens, next_token.unsqueeze(0)], dim=-1)
                position_ids = torch.tensor([self.seq_lens[seq_id]], device=input_ids.device)
                # key_cache, value_cache 都为空值
                key_cache, value_cache = self.block_manager.kv_cache.key_cache, self.block_manager.kv_cache.value_cache


                logits = self.model(
                    input_ids=input_ids,
                    position_ids=position_ids, # 生成一个初始化的 position_ids tensor，后续会从模型权重里加载对应的真实的值
                    attention_mask=None,
                    use_cache=True,
                    is_prefill=False,
                    key_cache=key_cache,
                    value_cache=value_cache,
                    slot_mappings=new_slot_mappings,
                    block_tables=paged_attention_block_table,
                    seq_lens= torch.tensor([self.seq_lens[seq_id]], dtype=torch.int32, device=input_ids.device),
                    max_seq_len=self.block_manager.kv_cache.max_blocks_per_seq * self.block_manager.block_size,
                )
                self.last_logits[seq_id] = logits[:, -1, :]
                self.seq_lens[seq_id] += 1

                # 检查是否已经结束 token，否则入 model，继续 next token
                if next_token.item() != self.model.config.eos_token_id and self.seq_lens[seq_id] < self.max_length:
                    print(f"seq {seq_id} is not processed done by GPT2 model, continue next token...")
                    self.queue.put((self.active_seqs[seq_id], seq_id))
                else:
                    print(f"seq {seq_id} is completed or reach max length {self.max_length}, final length is {self.seq_lens[seq_id]}")
                    self.remove_from_active(seq_id)

            except RuntimeError as e:

                if "CUDA out of memory" in str(e):
                    self.handle_out_of_memory([seq_id])
                else:
                    raise e


    def sample_next_token(self, seq_id: int) -> torch.Tensor:
        logits = self.last_logits[seq_id]
        temperature = 1.0 # 0.8
        logits = logits / temperature
        top_k = 50

        top_k_logits, top_k_indices = torch.topk(logits, top_k)
        # 根据 Transformer 架构图，attention_output 后还需要经过 Linear -> Softmax(在 scheduler 代码里)，才是 output probabilities
        probs = F.softmax(top_k_logits, dim=-1)
        next_token_index = torch.multinomial(probs, num_samples=1)
        next_token = top_k_indices[0, next_token_index[0]]
        return next_token

    def generate_seq_id(self) -> int:
        return int(time.time() * 1000000)



def generate_triangular_mask(batch_size, num_heads, seq_len):
    # Create an upper triangular matrix with -inf, including the diagonal
    # 上三角
    upper_triangular = torch.triu(torch.full((seq_len, seq_len), float('-inf'), dtype=torch.float16), diagonal=1)
    # Expand the upper triangular matrix to match the desired shape
    mask = upper_triangular.unsqueeze(0).unsqueeze(0)  # shape (1, 1, seq_len, seq_len)
    mask = mask.expand(batch_size, num_heads, seq_len, seq_len)  # shape (batch_size, num_heads, seq_len, seq_len)
    mask = mask.to("cuda", dtype=torch.float16)
    return mask
