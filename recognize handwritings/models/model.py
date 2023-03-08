import torch
from torch import nn

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
        x = x.view(x.size(0), -1)
        x = self.linear(x)
        return x