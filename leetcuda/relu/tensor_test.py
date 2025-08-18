


import torch

# 定义张量的长度
length = 10
# 创建一个值为-1和1相间隔的列表
values = [(-1) ** i for i in range(length)]
# 将列表转换为张量
tensor = torch.tensor(values)
print(tensor)


# 定义张量的长度
length = 10
# 生成一个从0到length-1的张量
indices = torch.arange(length)
# 使用 torch.where 根据索引的奇偶性设置值
tensor = torch.where(indices % 2 == 0, 1, -1)
print(tensor)


# 定义张量的长度
length = 10
# 创建两个张量，一个全是1，一个全是-1
ones = torch.ones(length // 2)
neg_ones = -torch.ones(length // 2)
# 将两个张量堆叠起来
stacked = torch.stack((ones, neg_ones), dim=1)
# 将堆叠后的张量展平
tensor = stacked.flatten()
print(tensor)

