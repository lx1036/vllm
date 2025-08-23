

# 激活函数的作用就是在线性函数的基础上增加了非线性。

# batch_size: 设置每个batch样本的大小



import pandas as pd
from torch.utils.data import DataLoader,Dataset
import torch


pd.set_option('display.max_columns', None)#打印时显示所有列。

# 从CSV文件读取数据（确保你有正确的路径）
df = pd.read_csv(r"./train.csv")

# 去除不需要的列
df = df.drop(columns=["PassengerId", "Name", "Ticket", "Cabin"])
# 去除 Age 缺失的样本
df = df.dropna(subset=["Age"])
# 对 Sex 和 Embarked 做独热编码
df = pd.get_dummies(df, columns=["Sex", "Embarked"],dtype=int)
# print(df.head(10))

class TitanicDataset(Dataset):
    def __init__(self, file_path):
        self.file_path = file_path
        self.mean = {
            "Pclass": 2.236695,
            "Age": 29.699118,
            "SibSp": 0.512605,
            "Parch": 0.431373,
            "Fare": 34.694514,
            "Sex_female": 0.365546,
            "Sex_male": 0.634454,
            "Embarked_C": 0.182073,
            "Embarked_Q": 0.039216,
            "Embarked_S": 0.775910
        }

        self.std = {
            "Pclass": 0.838250,
            "Age": 14.526497,
            "SibSp": 0.929783,
            "Parch": 0.853289,
            "Fare": 52.918930,
            "Sex_female": 0.481921,
            "Sex_male": 0.481921,
            "Embarked_C": 0.386175,
            "Embarked_Q": 0.194244,
            "Embarked_S": 0.417274
        }

        self.data = self._load_data()
        self.feature_size = len(self.data.columns) - 1

    def _load_data(self):
        df = pd.read_csv(self.file_path)
        df = df.drop(columns=["PassengerId", "Name", "Ticket", "Cabin"]) ##删除不用的列
        df = df.dropna(subset=["Age"])##删除Age有缺失的行
        df = pd.get_dummies(df, columns=["Sex", "Embarked"], dtype=int)##进行one-hot编码

        ##进行数据的标准化
        base_features = ["Pclass", "Age", "SibSp", "Parch", "Fare"]
        for i in range(len(base_features)):
            df[base_features[i]] = (df[base_features[i]] - self.mean[base_features[i]]) / self.std[base_features[i]]
        return df

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        features = self.data.drop(columns=["Survived"]).iloc[idx].values
        label = self.data["Survived"].iloc[idx]
        return torch.tensor(features, dtype=torch.float32), torch.tensor(label, dtype=torch.float32)

dataset = TitanicDataset(r"./train.csv")
dataloader = DataLoader(dataset, batch_size=32, shuffle=True)
for inputs, labels in dataloader:
    print(inputs.shape, labels.shape) # torch.Size([32, 10]) torch.Size([32])
    break
