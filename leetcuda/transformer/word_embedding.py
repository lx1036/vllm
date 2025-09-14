import math

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn



class Embedding(nn.Module):
    def __init__(self, vocab_size, d_model):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, d_model)


    def forward(self, x):
        return self.embedding(x)


input = torch.LongTensor([[5,2,0,1]]) # (1,4) 1行4列
# input = torch.Tensor([[5,2,0,1]]) # (1,4) 1行4列
print(input.shape)
src_vocab_size = 10
d_model = 512
word_emb = Embedding(src_vocab_size, d_model)
input_word_emb = word_emb(input)
print(input_word_emb, input_word_emb.shape)

'''
torch.Size([1, 4])w
tensor([[[-0.2633,  1.1539, -0.2080,  ...,  0.5939,  0.3132,  0.8706],
         [ 0.2300,  0.0186,  0.3298,  ..., -0.7758,  0.7763, -0.0650],
         [ 0.7271, -0.3776, -1.2280,  ...,  0.0627, -1.1944, -1.6579],
         [-0.1251, -1.2290,  0.2944,  ..., -0.7864,  0.5436,  0.2702]]],
       grad_fn=<EmbeddingBackward0>) torch.Size([1, 4, 512])
'''

input_word_emb = input_word_emb.transpose(0, 1)
print(input_word_emb, input_word_emb.shape)
'''
tensor([[[-0.7242,  2.6933,  0.3860,  ...,  0.9094,  1.3673, -1.0898]],

        [[-1.2536, -0.3182,  1.0572,  ...,  1.1498, -0.0227, -1.2943]],

        [[-0.3667,  0.2051,  1.5526,  ...,  0.5412, -0.5051, -0.1216]],

        [[-0.3450,  0.7312, -0.1419,  ...,  0.4379, -1.2209, -0.8523]]],
       grad_fn=<TransposeBackward0>) torch.Size([4, 1, 512])
'''

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, drop_out=0.1, max_len=5000):
        super().__init__()
        self.DropOut = nn.Dropout(p=drop_out)
        pe = torch.zeros(max_len, d_model) # (max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1) # (max_len,1),每个元素值取值范围在[0,max_len]，表示位置
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0)/d_model))
        # 偶数位置使用sin函数,奇数位置用cos函数
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0,1) # 矩阵转置
        self.register_buffer('pe', pe)
        print(f"pe: {pe}, {pe.shape}")

    def forward(self, x):
        # x: [seq_len, batch_size, d_model]
        pos = self.pe[:x.size(0), :, :]
        print(f"pos: {pos}, {pos.shape}")
        '''
        pos: tensor([[[ 0.0000e+00,  1.0000e+00,  0.0000e+00,  ...,  1.0000e+00,
           0.0000e+00,  1.0000e+00]],

        [[ 8.4147e-01,  5.4030e-01,  8.2186e-01,  ...,  1.0000e+00,
           1.0366e-04,  1.0000e+00]],

        [[ 9.0930e-01, -4.1615e-01,  9.3641e-01,  ...,  1.0000e+00,
           2.0733e-04,  1.0000e+00]],

        [[ 1.4112e-01, -9.8999e-01,  2.4509e-01,  ...,  1.0000e+00,
           3.1099e-04,  1.0000e+00]]]), torch.Size([4, 1, 512])
        '''

        # 前10个位置编码，可视化前6个维度和最后一个维度
        pos2 = self.pe[:10, :, :]
        pos_encoding = pos2.squeeze(1).numpy()
        plt.figure(figsize=(10, 6))
        positions = np.arange(0, 10)
        plt.plot(positions, pos_encoding[:, 0], label="Dimension 0")
        plt.plot(positions, pos_encoding[:, 1], label="Dimension 1")
        plt.plot(positions, pos_encoding[:, 2], label="Dimension 2")
        plt.plot(positions, pos_encoding[:, 3], label="Dimension 3")
        plt.plot(positions, pos_encoding[:, 4], label="Dimension 4")
        plt.plot(positions, pos_encoding[:, 5], label="Dimension 5")
        plt.plot(positions, pos_encoding[:, 511], label="Dimension 511")
        plt.xlabel("Position")
        plt.ylabel("Value")
        plt.title("Positional Encoding")
        plt.legend()
        plt.show()

        # 词编码加上位置编码，就是 transformer 的输入数据
        x = x + pos
        return x

pe = PositionalEncoding(d_model)
input_transformer = pe(input_word_emb).transpose(0,1)
print(f"input_transformer: {input_transformer}, {input_transformer.shape}")
'''
input_transformer: tensor([[[ 0.3789,  2.0902,  0.8227,  ...,  0.4997,  0.2827,  0.3053],
         [ 1.8672,  0.9527,  0.8931,  ..., -0.2709,  0.6905,  1.1348],
         [ 2.4843, -0.0339,  2.0067,  ...,  1.9731, -0.7038,  1.6767],
         [ 0.5222, -0.7663,  1.2045,  ..., -0.0388,  0.4564,  0.2593]]],
       grad_fn=<TransposeBackward0>), torch.Size([1, 4, 512])
'''



torch.Tensor(1, 1, 512).stride(1)
