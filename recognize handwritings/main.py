# 获取数据
# 建立模型
# 训练脚本
# 推理脚本

import pandas as pd
from torchvision import datasets
from torch.utils.data import DataLoader


# 读取手写数据集
train_data = datasets.MNIST(root="./MNIST",
                            train=True,
                            download=False)

for i in train_data:
    print(i)