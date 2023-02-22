# 获取数据
# 建立模型
# 训练脚本
# 推理脚本

import pandas as pd
import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from torch import nn


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


# 构建CNN模型
class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.conv2d = torch.nn.Conv2d(in_channels=1, out_channels=1, kernel_size=3)
        self.maxpool = torch.nn.MaxPool2d(kernel_size=3, stride=1)
        self.linear = torch.nn.Linear(24 * 24, 10)

    def forward(self, x):
        x = self.conv2d(x)
        x = self.maxpool(x)
        print(x.size())
        x = x.reshape(1, -1)
        x = self.linear(x)
        return x


# 训练模型脚本
cnn = CNN()
loss_function = torch.nn.MSELoss()
optimizer = torch.optim.SGD(cnn.parameters(), lr=0.1)

for epoch in range(100):
    for images in train_loader:
        x = images[0]
        y_label = images[1]
        y = cnn.forward(x)
        loss = loss_function(y, y_label)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    print("完成第{}轮训练，loss为{}".format(epoch, loss))



