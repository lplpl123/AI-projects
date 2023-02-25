# 建立模型
# 推理脚本
import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from model import CNN

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


# 训练模型脚本
cnn = CNN()
loss_function = torch.nn.CrossEntropyLoss()
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



