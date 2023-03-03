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
test_loader = DataLoader(dataset=test_data, batch_size=5000, shuffle=True)


# 训练模型脚本
cnn = CNN()
loss_function = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(cnn.parameters(), lr=0.1)

def train():
    for epoch in range(1): # todo 临时调整训练参数
        for images in train_loader:
            x = images[0]
            y_label = images[1]
            y = cnn.forward(x)
            loss = loss_function(y, y_label)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        print("完成第{}轮训练，loss为{}".format(epoch, loss))

# 计算精度
def test():
    correct = 0
    for images in test_loader:
        x = images[0]
        y_label = images[1]
        y = cnn.forward(x)
        pred = y.argmax(dim=1)
        match_lst = pred.eq(y_label)
        for ele in match_lst:
            if ele == True:
                correct += 1
    accuracy = correct / 10000 # todo 广义化
    print("accuracy: " + str(accuracy))

train()
torch.save(cnn, "./models/cnn.pth")
torch.save(cnn.state_dict(), './models/cnn.params')
test()


