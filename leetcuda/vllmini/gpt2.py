from typing import Optional

import torch.nn as nn
import torch
from transformers import GPT2Config


# 注意力机制
class GPT2Attention(nn.Module):
    def __init__(self, config: GPT2Config):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.num_heads = config.num_attention_heads # 12
        self.head_dim = self.hidden_size // self.num_heads # 64=768/12
        self.scale = self.head_dim**-0.5 # 1/sqrt(dk) 平方根再取倒数
        self.linear = nn.Linear(self.hidden_size, 3 * self.hidden_size, bias=True)
        self.projection = nn.Linear(self.hidden_size, self.hidden_size, bias=True)


    def forward(self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        use_cache: bool = True,
        is_prefill: bool = True,
        key_cache: Optional[torch.Tensor] = None,
        value_cache: Optional[torch.Tensor] = None,
        slot_mapping: Optional[torch.Tensor] = None,
        block_tables: Optional[torch.Tensor] = None,
        seq_lens: Optional[torch.Tensor] = None,
        max_seq_len: Optional[int] = None,
    ):
        batch_size, seq_len, _ = hidden_states.shape # [batch_size=1, seq_len=4("我 喜欢 吃 酸菜鱼"), hidden_size=768]
        qkv = self.linear(hidden_states)
        q, k ,v = qkv.split(self.hidden_size, dim=-1)
        # Reshape q, k, v to [batch_size * seq_len, num_heads, head_dim]
        q = q.view(-1, self.num_heads, self.head_dim)
        k = k.view(-1, self.num_heads, self.head_dim)
        v = v.view(-1, self.num_heads, self.head_dim)

        # k=[batch_size * seq_len, num_heads, head_dim]
        cache_ops.reshape_and_cache(k, v, key_cache, value_cache, slot_mapping, "auto", 1.0)

        if is_prefill:
            # [batch_size, seq_len, num_heads, head_dim]
            q =q.view(batch_size, seq_len, self.num_heads, self.head_dim)
            k =k.view(batch_size, seq_len, self.num_heads, self.head_dim)
            v =v.view(batch_size, seq_len, self.num_heads, self.head_dim)

            # [batch_size, num_heads, seq_len, head_dim]
            q = q.transpose(1, 2)
            k = k.transpose(1, 2)
            v = v.transpose(1, 2)

            attention_output = self.attention(q, k, v, attention_mask)
            # [batch_size, seq_len, num_heads, head_dim] -> [batch_size, seq_len, hidden_size=num_heads*head_dim]
            attention_output = attention_output.transpose(1, 2).contiguous().view(batch_size, seq_len, self.hidden_size)

        else:

            # PagedAttention 里算 Attention(Q,K,V) = softmax(QK^T/sqrt(dk))V
            # [batch_size * seq_len, num_heads, head_dim]
            attention_output = torch.empty_like(q)

        attention_output = self.projection(attention_output)
        if use_cache:
            return attention_output, (k ,v) # 这里为啥是 (k,v)
        return attention_output, None

    # Pytorch函数计算 Attention(Q,K,V) = softmax(QK^T/sqrt(dk))V
    def attention(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, attention_mask: Optional[torch.Tensor] = None,):
        # [batch_size, num_heads, seq_len, head_dim] * [batch_size, num_heads, head_dim, seq_len]
        attention_weights = torch.matmul(q, k.transpose(-1, -2)) * self.scale
        if attention_mask is not None:
            # 直接相加，来覆盖之后的 token
            attention_weights = attention_weights + attention_mask
        attention_weights = nn.functional.softmax(attention_weights, dim=-1)
        return torch.matmul(attention_weights, v)


# 前馈神经网络
class GPT2MLP(nn.Module):
    def __init__(self):
        super().__init__()


# 多头注意力机制
class GPT2Block(nn.Module):
    def __init__(self, config: GPT2Config):
        super().__init__()

        self.attention = GPT2Attention(config)

    def forward(self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        use_cache: bool = True,
        is_prefill: bool = True,
        key_cache: Optional[torch.Tensor] = None,
        value_cache: Optional[torch.Tensor] = None,
        slot_mapping: Optional[torch.Tensor] = None,
        block_table: Optional[torch.Tensor] = None,
        seq_lens: Optional[torch.Tensor] = None,
        max_seq_len: Optional[int] = None,
    ):

        attention_outputs = self.attention(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            use_cache=use_cache,
            is_prefill=is_prefill,
            key_cache=key_cache,
            value_cache=value_cache,
            slot_mapping=slot_mapping,
            block_table=block_table,
            seq_lens=seq_lens,
            max_seq_len=max_seq_len,
        )



class GPT2Model(nn.Module):
    def __init__(self, config: GPT2Config):
        super().__init__()
        self.config = config
        self.word_token_embeddings = nn.Embedding(config.vocab_size, config.hidden_size)
        self.word_position_embedding = nn.Embedding(config.max_position_embeddings, config.hidden_size)
        self.blocks = nn.ModuleList([GPT2Block(config) for _ in range(config.num_hidden_layers)])

    def forward(self,
        input_ids: torch.Tensor,
        position_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        use_cache: bool = True,
        is_prefill: bool = True,
        key_cache: Optional[torch.Tensor] = None,
        value_cache: Optional[torch.Tensor] = None,
        slot_mappings: Optional[torch.Tensor] = None,
        block_tables: Optional[torch.Tensor] = None,
        seq_lens: Optional[torch.Tensor] = None,
        max_seq_len: Optional[int] = None,
    ):
        # 词矩阵和位置矩阵相加，可得输入矩阵
        hidden_states = self.word_token_embeddings(input_ids) + self.word_position_embedding(position_ids)

        for i, block in enumerate(self.blocks):
            slot_mapping = slot_mappings[i] if slot_mappings is not None else None
            block_table = block_tables[i] if block_tables is not None else None
            outputs = block(
                hidden_states=hidden_states,
                attention_mask=attention_mask,
                use_cache=use_cache,
                is_prefill=is_prefill,
                key_cache=key_cache,
                value_cache=value_cache,
                slot_mapping=slot_mapping,
                block_table=block_table,
                seq_lens=seq_lens,
                max_seq_len=max_seq_len,
            )



class GPT2LMHeadModel(nn.Module):
    def __init__(self, config: GPT2Config):
        super().__init__()
        self.config = config
        self.transformer = GPT2Model(config)
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False) # [512, vocab_size=50257]

    def forward(self,
        input_ids: torch.Tensor,
        position_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        use_cache: bool = True,
        is_prefill: bool = True,
        key_cache: Optional[torch.Tensor] = None,
        value_cache: Optional[torch.Tensor] = None,
        slot_mappings: Optional[torch.Tensor] = None,
        block_tables: Optional[torch.Tensor] = None,
        seq_lens: Optional[torch.Tensor] = None,
        max_seq_len: Optional[int] = None,
    ):
        transformer_outputs = self.transformer(
            input_ids=input_ids,
            position_ids=position_ids,
            attention_mask=attention_mask,
            use_cache=use_cache,
            is_prefill=is_prefill,
            key_cache=key_cache,
            value_cache=value_cache,
            slot_mappings=slot_mappings,
            block_tables=block_tables,
            seq_lens=seq_lens,
            max_seq_len=max_seq_len,
        )
        hidden_states = transformer_outputs[0]
        # 线性变化为 词表
        logits = self.lm_head(hidden_states)
        return logits, transformer_outputs[1] if use_cache else None



