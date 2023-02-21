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
train_loader = DataLoader(dataset=train_data, batch_size=10000, shuffle=True)
for loader in train_loader:
    print(loader[1].requires_grad)


class CNN():
    def __init__(self):
        self.conv2d = torch.nn.Conv2d(in_channels=1, out_channels=1, kernel_size=3)
        self.maxpool = torch.nn.MaxPool2d(kernel_size=3, stride=1)
        self.linear = torch.nn.Linear(24 * 24, 10)
        self.loss_function = torch.nn.MSELoss()
        self.optimizer = None # todo

    def forward(self, x):
        x = self.conv2d(x)
        x = self.maxpool(x)
        x = x.reshape(1, -1)
        x = self.linear(x)
        return x

    def train(self):
        for epoch in range(100):
            for loader in train_loader:
                x = loader[0]
                y_label = loader[1]
                y = self.forward(x)
                loss = self.loss_function(y, y_label)

                loss.backward()



