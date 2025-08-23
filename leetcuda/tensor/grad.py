
# https://www.rethink.fun/chapter6/%E7%94%A8PyTorch%E5%AE%9E%E7%8E%B0%E7%BA%BF%E6%80%A7%E5%9B%9E%E5%BD%92.html

import torch
from torch.utils.tensorboard import SummaryWriter

x = torch.tensor(1.0, requires_grad=True)
y = torch.tensor(1.0, requires_grad=True)
print(x, y)
# tensor(1., requires_grad=True) tensor(1., requires_grad=True)
v = 3*x+4*y
u = torch.square(v)
z = torch.log(u)

z.backward() # 反向传播求梯度

print("x grad:", x.grad)
print("y grad:", y.grad)
'''
x grad: tensor(0.8571)
y grad: tensor(1.1429)
'''

print(torch.tensor([1,1,1]) + torch.tensor(10)) # tensor([11, 11, 11])

# bias 偏置
# 确保CUDA可用
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# 生成数据
inputs = torch.rand(100, 3) # 1.随机生成100条数据，每条数据都有3个feature, 里边每个元素的值都是0-1之间
weights = torch.tensor([[1.1], [2.2], [3.3]]) #预设的权重
print(weights.shape) # torch.Size([3, 1])
bias = torch.tensor(4.4) #预设的bias
# 100*3 @ 3*1=100*1
targets = inputs @ weights + bias + 0.1*torch.randn(100, 1) #增加一些误差，模拟真实情况 # 对随机生成的inputs经过线性变化，再加上一些小的随机误差
# 生成对应的targets作为label值
print(targets, targets.shape) # torch.Size([100, 1])


# 创建一个SummaryWriter实例
writer = SummaryWriter(log_dir="./tmp/")

# 初始化参数时直接放在CUDA上，并启用梯度追踪
w = torch.rand((3, 1), requires_grad=True, device=device) # weight
b = torch.rand((1,), requires_grad=True, device=device) # bias
print("训练前的权重 w:", w)
print("训练前的偏置 b:", b)
'''
训练前的权重 w: tensor([[0.9947],
        [0.9330],
        [0.3104]], requires_grad=True)
训练前的偏置 b: tensor([0.1098], requires_grad=True)
'''

# 将数据移至相同设备
inputs = inputs.to(device)
targets = targets.to(device)

#设置超参数
epoch = 10000 # 超参数迭代次数epoch
lr = 0.003 # 学习率lr
for i in range(epoch):
    outputs = inputs @ w + b

    # 等于 Sum[0,j](outputs - targets)^2/j
    loss = torch.mean(torch.square(outputs - targets)) # loss 函数
    # print("loss:", loss.item())

    #记录loss，三个参数分别：tag，loss值，第几步
    writer.add_scalar("loss/train", loss.item(), i)

    loss.backward()

    with torch.no_grad(): #下边的计算不需要跟踪梯度
        w -= lr * w.grad
        b -= lr * b.grad

    # 清零梯度
    w.grad.zero_()
    b.grad.zero_()


print("训练后的权重 w:", w)
print("训练后的偏置 b:", b)

'''
训练后的权重 w: tensor([[1.0938],
        [2.2847],
        [3.3318]], requires_grad=True)
训练后的偏置 b: tensor([4.3580], requires_grad=True)
'''



'''
pip3 install tensorboard
python3 grad.py
tensorboard --logdir=./tmp/
'''
