import time


# https://zhuanlan.zhihu.com/p/1933089461601833925
# https://www.rethink.fun/chapter6/Tensor.html

# 《使用张量表征真实数据》：https://weread.qq.com/web/reader/73d32f9072922e7d73d5a39kd9d320f022ed9d4f495e456


import torch
import numpy as np

# 1D Tensor
t1 = torch.tensor([1, 2, 3])
print(t1)

# 2D Tensor
t2 = torch.tensor([[1, 2, 3], [4, 5, 6]])
print(t2)

# 3D Tensor
t3 = torch.tensor([[[1, 2], [3, 4]], [[5, 6], [7, 8]]])
print(t3)

# 从 NumPy 创建 Tensor
arr = np.array([1, 2, 3])
t_np = torch.tensor(arr)
print(t_np)


t1 = torch.tensor((2,2),dtype=torch.float32)
print(t1)
# 整数型 torch.uint8、torch.int32、torch.int64
# 浮点型 torch.float16、torch.bfloat16、 torch.float32、torch.float64, 其中torch.float32为默认的浮点数据类型
# torch.float32称为全精度，torch.float16/torch.bfloat16称为半精度。一般情况下模型的训练是在全精度下进行的。
# 布尔型 torch.bool

x = torch.tensor([1, 2, 3, 4, 5])
mask = x > 2  # 生成一个布尔掩码
print(mask) # tensor([False, False,  True,  True,  True])

# 用布尔掩码选出大于 2 的值
filtered_x = x[mask]
print(filtered_x) # tensor([3, 4, 5])

# 用布尔掩码选出大于 2 的值,并赋值为0
x[mask]=0
print(x) # tensor([1, 2, 0, 0, 0])

shape = (2,3)
rand_tensor = torch.rand(shape) # 生成一个从[0,1]均匀抽样的tensor。
print(rand_tensor)
'''
tensor([[0.8881, 0.9755, 0.0103],
        [0.0803, 0.7321, 0.9171]])
'''
randn_tensor = torch.randn(shape) # 生成一个从标准正态分布抽样的tensor。
print(randn_tensor)
'''
tensor([[-0.2972,  0.8817, -0.7130],
        [-0.6554, -0.5967,  2.1065]])
'''
ones_tensor = torch.ones(shape) #生成一个值全为1的tensor。
print(ones_tensor)
zeros_tensor = torch.zeros(shape) # 生成一个值全为0的tensor。
print(zeros_tensor)
twos_tensor = torch.full(shape, 2) #  生成一个值全为2的tensor。
print(twos_tensor)

# 我们知道一个tensor有几个常用的关键属性，第一个是tensor的形状，第二个是tensor内元素的类型，第三个是tensor的设备
# tensor还有一个重要属性，requires_grad是否需要计算梯度
tensor = torch.rand(3,4)
print(f"Shape of tensor: {tensor.shape}")
print(f"Datatype of tensor: {tensor.dtype}")
print(f"Device tensor is stored on: {tensor.device}")
print(f"requires_grad: {tensor.requires_grad}")
'''
Shape of tensor: torch.Size([3, 4])
Datatype of tensor: torch.float32
Device tensor is stored on: cpu
requires_grad: False
'''

x = torch.randn(4,4) #  生成一个形状为4x4的随机矩阵。
print(x)
'''
tensor([[-0.7768,  1.6071,  0.3838,  0.4237],
        [ 0.0097,  0.0820,  0.7840, -0.2881],
        [ 0.2989, -0.3341,  0.1803,  0.8468],
        [ 0.2591,  0.1168,  0.2760,  0.9262]])
'''
x = x.reshape(2,8) # 通过reshape操作，可以将4x4的矩阵改变为2x8的矩阵。
print(x)
'''
tensor([[-0.7768,  1.6071,  0.3838,  0.4237,  0.0097,  0.0820,  0.7840, -0.2881],
        [ 0.2989, -0.3341,  0.1803,  0.8468,  0.2591,  0.1168,  0.2760,  0.9262]])
'''

x = torch.tensor([[1, 2, 3], [4, 5, 6]])
x_reshape = x.reshape(3,2)
x_transpose = x.permute(1,0) #交换第0个和第1个维度。对于二维矩阵就是行列互换，进行转置。
print("reshape:",x_reshape)
print("permute:",x_transpose)
'''
reshape: tensor([[1, 2],
        [3, 4],
        [5, 6]])
permute: tensor([[1, 4],
        [2, 5],
        [3, 6]])
'''

x = torch.tensor([[1,2,3],[4,5,6]])
#扩展第0维
x_0 = x.unsqueeze(0)
print(x_0.shape,x_0)
#扩展第1维
x_1 = x.unsqueeze(1)
print(x_1.shape,x_1)
#扩展第2维
x_2 = x.unsqueeze(2)
print(x_2.shape,x_2)
'''
torch.Size([1, 2, 3]) tensor([[[1, 2, 3],
         [4, 5, 6]]])
torch.Size([2, 1, 3]) tensor([[[1, 2, 3]],

        [[4, 5, 6]]])
torch.Size([2, 3, 1]) tensor([[[1],
         [2],
         [3]],

        [[4],
         [5],
         [6]]])
'''

x = torch.ones((1,1,3))
print(x.shape, x)
y = x.squeeze(dim=0)
print(y.shape, y)
z = x.squeeze()
print(z.shape, z)
'''
torch.Size([1, 1, 3]) tensor([[[1., 1., 1.]]])
torch.Size([1, 3]) tensor([[1., 1., 1.]])
torch.Size([3]) tensor([1., 1., 1.])
'''

a = torch.ones((2,3))
b = torch.ones((2,3))
print(a + b)  # 加法
print(a - b)  # 减法
print(a * b)  # 逐元素乘法
print(a / b)  # 逐元素除法
print(a @ b.t())  # 矩阵乘法 b.t 为 b 矩阵的转置
'''
tensor([[2., 2., 2.],
        [2., 2., 2.]])
tensor([[0., 0., 0.],
        [0., 0., 0.]])
tensor([[1., 1., 1.], # 逐元素乘法 1*1
        [1., 1., 1.]])
tensor([[1., 1., 1.],
        [1., 1., 1.]])
tensor([[3., 3.], # 2*3 @ 3*2
        [3., 3.]])
'''

t1 = torch.randn((3,2))
print(t1)
t2 = t1 + 1 # 广播机制
print(t2)
'''
tensor([[-0.7673,  1.0162],
        [ 1.5114,  0.3365],
        [-0.7176, -1.2534]])
tensor([[ 0.2327,  2.0162],
        [ 2.5114,  1.3365],
        [ 0.2824, -0.2534]])
'''

t1 = torch.ones((3,2))
t2 = torch.ones(2)
t3 = t1 + t2 # 广播机制
print(t1, t1.shape)
print(t2, t2.shape)
print(t3, t3.shape)
'''
tensor([[1., 1.],
        [1., 1.],
        [1., 1.]]) torch.Size([3, 2])
tensor([1., 1.]) torch.Size([2])
tensor([[2., 2.],
        [2., 2.],
        [2., 2.]]) torch.Size([3, 2])
'''

'''
你默认创建的tensor都是在CPU/内存上的。你有两种方法让tensor转移到GPU/显存上。
1、创建时，设定tensor的设备为“cuda”。
2、将cpu上的tensor通过to("cuda")方法转移到GPU上。
'''

x = torch.randn(1, 2)
print(x)
print(torch.mean(x)) # 求平均值
'''
tensor([[-0.7389, -0.4926]])
tensor(-0.6157)
'''
exit(0)



'''
################################  利用GPU加速计算  #####################################
'''
# 确保 GPU 可用
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# 生成随机矩阵
size = 10000  # 矩阵大小
A_cpu = torch.rand(size, size) # 默认在CPU上创建tensor
B_cpu = torch.rand(size, size)

start_cpu = time.time()
C_cpu = torch.mm(A_cpu, B_cpu)  # 矩阵乘法
end_cpu = time.time()
cpu_time = end_cpu - start_cpu

# 在 GPU 上计算
A_gpu = A_cpu.to(device) # 将tensor转移到GPU上
B_gpu = B_cpu.to(device)

start_gpu = time.time()
C_gpu = torch.mm(A_gpu, B_gpu)
torch.cuda.synchronize()  # 确保GPU计算完成
end_gpu = time.time()
gpu_time = end_gpu - start_gpu

print(f"CPU time: {cpu_time:.6f} sec")
if torch.cuda.is_available():
    print(f"GPU time: {gpu_time:.6f} sec")
else:
    print("GPU not available, skipping GPU test.")

'''
CPU time: 1.415197 sec
GPU time: 0.099745 sec
'''


'''
(1)深度学习里，经常用两个向量之间的夹角的余弦值来表示两个向量的相似度。
比如在深度学习里，模型学习到的人脸特征都是用向量来表示，最终如何判断两个人脸向量是否代表同一个人，就是通过计算这两个向量之间的余弦值来判断。如果余弦值接近1。也就是两个向量之间夹角接近0，则认为这两个向量相似，也就是两个人脸照片相似。



'''



