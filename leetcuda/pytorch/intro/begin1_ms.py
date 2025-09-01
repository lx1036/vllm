import mindspore
from mindspore import Tensor, nn
from mindspore.dataset import MnistDataset, vision
from mindspore.dataset.vision import transforms

# 【昇思MindSpore：深度学习原理与实践】: https://www.hiascend.com/developer/courses/detail/1938153539479527425


tensor_0d = Tensor(24)
tensor_1d = Tensor([1,2,3])
tensor_2d = Tensor([(1,2,3), (4,5,6)])
tensor_3d = Tensor([[(1,2,3), (4,5,6)], [(7,8,9), (10,11,12)], [(13,14,15), (16,17,18)]])
print(tensor_0d, tensor_1d, tensor_2d, tensor_3d)
print(tensor_0d.shape, tensor_1d.shape, tensor_2d.shape, tensor_3d.shape)
print(tensor_0d.dtype, tensor_1d.dtype, tensor_2d.dtype, tensor_3d.dtype)
print(tensor_0d.ndim, tensor_1d.ndim, tensor_2d.ndim, tensor_3d.ndim)
# print(tensor_0d.size(), tensor_1d.size(), tensor_2d.size(), tensor_3d.size())
# print(tensor_0d.min(), tensor_1d.min(), tensor_2d.min(), tensor_3d.min())
# print(tensor_0d.max(), tensor_1d.max(), tensor_2d.max(), tensor_3d.max())

'''
24 [1 2 3] [[1 2 3]
 [4 5 6]] [[[ 1  2  3]
  [ 4  5  6]]

 [[ 7  8  9]
  [10 11 12]]

 [[13 14 15]
  [16 17 18]]]
() (3,) (2, 3) (3, 2, 3)
Int64 Int64 Int64 Int64
0 1 2 3
'''

# pip3 install download
from download import download

# 1. 下载数据集
# 不会检查已经下载的文件
# url = "https://mindspore-website.obs.cn-north-4.myhuaweicloud.com/notebook/datasets/MNIST_Data.zip"
# path = download(url, "./tmp/", kind="zip", replace=True)

# 2. 加载数据集
train_dataset = MnistDataset("./tmp/MNIST_Data/train")
test_dataset = MnistDataset("./tmp/MNIST_Data/test")

image, label = next(train_dataset.create_tuple_iterator())
print(type(image))
print(f"Shape of image[H W C]: {image.shape}")
print(label.shape)
'''
<class 'mindspore.common.tensor.Tensor'>
Shape of image[H W C]: (28, 28, 1)
()
'''

# 3. 数据预处理
print(train_dataset.get_col_names())
'''
['image', 'label']
'''

def datapipe(dataset, batch_size):
    image_transforms = [
        vision.Rescale(1.0/255.0, 0),
        vision.Normalize(mean=(0.1307,), std=(0.3081,)),
        vision.HWC2CHW(),
    ]
    label_transform = transforms.TypeCast(mindspore.int32)
    dataset = dataset.map(image_transforms, "image")
    dataset = dataset.map(label_transform, "label")
    dataset = dataset.batch(batch_size, drop_remainder=True)
    return dataset

train_dataset = datapipe(train_dataset, 64)
test_dataset = datapipe(test_dataset, 64)

for data in train_dataset.create_dict_iterator():
    print(data["image"].shape, data["label"].shape)
    '''
    (64, 1, 28, 28) (64,)
    '''
    break

image, label = next(train_dataset.create_tuple_iterator())
print(image.shape)
print(label.shape)
'''
(64, 1, 28, 28)
(64,)
'''

flatten = nn.Flatten()
print(flatten(image).shape)
'''
1*28*28=784
(64, 784)
'''

class Network(nn.Cell):
    def __init__(self):
        super().__init__()
        # flatten 将数据展开，经常被放在卷积层和全连接层之间，将卷积层输出的特征图转换为向量序列形式，方便后续全连接层处理。
        self.flatten = nn.Flatten()
        self.dense_relu_sequential = nn.SequentialCell(
            # 全连接层，使用权重和偏差对输入进行线性变换，输入tensor(64, 28*28)，所以in_channels=784
      nn.Dense(28 * 28, 512),# 输出为tensor(64*512)
            # 非线性激活函数，帮助神经网络学习各种复杂的特征。>0的保留，<0的截断为0
            nn.ReLU(),
            nn.Dense(512, 512),
            nn.ReLU(),
            nn.Dense(512, 10) # 最后是tensor(64*10)
        )

    # forward
    def construct(self, x):
        x = self.flatten(x)
        logits = self.dense_relu_sequential(x)
        return logits

model = Network()
print(model)
for param in model.get_parameters():
    print(param)
    # break
'''
Network(
  (flatten): Flatten()
  (dense_relu_sequential): SequentialCell(
    (0): Dense(input_channels=784, output_channels=512, has_bias=True)
    (1): ReLU()
    (2): Dense(input_channels=512, output_channels=512, has_bias=True)
    (3): ReLU()
    (4): Dense(input_channels=512, output_channels=10, has_bias=True)
  )
)

Parameter (name=dense_relu_sequential.0.weight, shape=(512, 784), dtype=Float32, requires_grad=True)
Parameter (name=dense_relu_sequential.0.bias, shape=(512,), dtype=Float32, requires_grad=True)
Parameter (name=dense_relu_sequential.2.weight, shape=(512, 512), dtype=Float32, requires_grad=True)
Parameter (name=dense_relu_sequential.2.bias, shape=(512,), dtype=Float32, requires_grad=True)
Parameter (name=dense_relu_sequential.4.weight, shape=(10, 512), dtype=Float32, requires_grad=True)
Parameter (name=dense_relu_sequential.4.bias, shape=(10,), dtype=Float32, requires_grad=True)
'''

'''
1.计算正向结果logits
2.计算logits与正确标签targets的loss
def forward_fn(data, label):
    logits = model(data)
    loss = loss_fn(logits, label)
    return loss, logits

手写数字识别任务属于多分类问题，适合使用交叉熵损失CrossEntropyLoss

3.反向传播获取梯度grad
4.更新grad到网络权重parameter

'''


'''
1. 词向量 VocabEmbedding 和 词位置向量 PositionalEmbedding
旋转位置编码（RotaryPositionalEmbedding）：绝对和相对信息的融合，用绝对位置编码来表征相对位置编码
'''





