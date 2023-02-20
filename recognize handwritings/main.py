# 获取数据
# 建立模型
# 训练脚本
# 推理脚本

import pandas as pd
import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader


# 把读取到的图片转换为tensor
transform = transforms.Compose([
    transforms.ToTensor()
])
# 读取手写数据集, 总共有60000张图片
train_data = datasets.MNIST(root="./MNIST",
                            train=True,
                            transform=transform,
                            download=False)
test_data = datasets.MNIST(root="./MNIST",
                           train=False,
                           transform=transform,
                           download = False)

class CNN():
    def __init__(self):
        self.conv2d = torch.nn.Conv2d(in_channels=1, out_channels=1, kernel_size=3)


