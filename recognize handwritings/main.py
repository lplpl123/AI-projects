# 获取数据
# 建立模型
# 训练脚本
# 推理脚本

import pandas as pd
from torchvision import datasets, transforms
from torch.utils.data import DataLoader


# 读取手写数据集, 总共有60000张图片
train_data = datasets.MNIST(root="./MNIST",
                            train=True,
                            download=False)
test_data = datasets.MNIST(root="./MNIST",
                           train=False,
                           download = False)