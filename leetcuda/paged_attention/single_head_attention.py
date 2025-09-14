


# https://zhuanlan.zhihu.com/p/26911261250


import torch
import torch.nn as nn
import torch.nn.functional as F

class SingleHeadAttention(nn.Module):
    def __init__(self, embed_dim):
        """
        单头注意力机制的初始化。
        :param embed_dim: 嵌入维度，Query、Key 和 Value 的维度
        """
        super(SingleHeadAttention, self).__init__()
        self.embed_dim = embed_dim

        # 定义线性层，将输入映射到 Query、Key 和 Value
        self.query_linear = nn.Linear(embed_dim, embed_dim)
        self.key_linear = nn.Linear(embed_dim, embed_dim)
        self.value_linear = nn.Linear(embed_dim, embed_dim)

        # 缩放因子，用于防止点积结果过大
        self.scale = torch.sqrt(torch.FloatTensor(embed_dim))

    def forward(self, query, key, value):
        """
        单头注意力的前向传播。
        :param query: 查询张量，形状为 $batch_size, seq_len_q, embed_dim]
        :param key: 键张量，形状为 $batch_size, seq_len_k, embed_dim]
        :param value: 值张量，形状为 $batch_size, seq_len_k, embed_dim]
        :return: 输出张量，形状为 $batch_size, seq_len_q, embed_dim]
        """
        # 将输入映射到 Query、Key 和 Value
        Q = self.query_linear(query)
        K = self.key_linear(key)
        V = self.value_linear(value)

        # 计算点积注意力分数
        attention_scores = torch.matmul(Q, K.transpose(-2, -1)) / self.scale

        # 应用 Softmax 函数，得到注意力权重
        attention_weights = F.softmax(attention_scores, dim=-1)

        # 加权求和，得到最终输出
        output = torch.matmul(attention_weights, V)

        return output, attention_weights


# 示例输入
# 假设我们有以下输入张量：
# - query: $batch_size, seq_len_q, embed_dim]
# - key: $batch_size, seq_len_k, embed_dim]
# - value: $batch_size, seq_len_k, embed_dim]
batch_size = 2
seq_len_q = 3 # query的序列长度 2,3,6
seq_len_k = 4 #k,v的序列长度，注意这里K、V是成对存在的 2,4,6
embed_dim = 6 # 假设embedding的维度为6

# 随机生成输入数据
query = torch.randn(batch_size, seq_len_q, embed_dim)
key = torch.randn(batch_size, seq_len_k, embed_dim)
value = torch.randn(batch_size, seq_len_k, embed_dim)

# 初始化单头注意力模块
attention = SingleHeadAttention(embed_dim)

# 前向传播
output, attention_weights = attention(query, key, value)

# 打印输出
print("Query:\n", query)
print("Key:\n", key)
print("Value:\n", value)
print("Output:\n", output)
print("Attention Weights:\n", attention_weights)

