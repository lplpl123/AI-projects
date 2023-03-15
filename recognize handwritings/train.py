# 推理脚本
import time
import csv
import torch
from config import params
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from models.model import CNN


# train function: to train the parameters
def train():
    for epoch in range(params.EPOCHES):
        for images in train_loader:
            x = images[0]
            y_label = images[1]
            y = cnn.forward(x)
            loss = loss_function(y, y_label)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        print("完成第{}轮训练，loss为{}".format(epoch, loss))
    return loss

# test function: to calculate the accuracy and loss
def test():
    correct = 0
    total = 0
    for images in test_loader:
        x = images[0]
        y_label = images[1]
        y = cnn.forward(x)
        pred = y.argmax(dim=1)
        total += y_label.size(0)
        correct += (pred == y_label).sum().item()
    accuracy = correct / total
    print("accuracy: " + str(accuracy))
    return accuracy

# 记录训练结果和参数
def record(train_loss, *values):
    write_lst = [time.ctime(), str(train_loss.item())]
    for ele in values:
        write_lst.append(ele)
    with open('./data/train_params_and_outputs/recordings.csv', 'a', encoding='utf-8') as file:
        writer = csv.writer(file)
        writer.writerow(write_lst)


if __name__ == '__main__':

    # 读取数据集的mean和std
    # 把读取到的图片转换为tensor
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(params.TOL_MEAN, params.TOL_STD)
    ])
    # 读取手写数据集, 总共有60000张图片
    train_data = datasets.MNIST(root="./data/MNIST",
                                train=True,
                                transform=transform,
                                download=False)
    test_data = datasets.MNIST(root="./data/MNIST",
                               train=False,
                               transform=transform,
                               download=False)
    train_loader = DataLoader(dataset=train_data, batch_size=params.TRAIN_BATCH_SIZE, shuffle=True)
    test_loader = DataLoader(dataset=test_data, batch_size=params.TEST_BATCH_SIZE, shuffle=True)

    # 训练模型脚本
    cnn = CNN()
    loss_function = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(cnn.parameters(), lr=params.LEARNING_RATE)
    start_time = time.time()
    train_loss = train()
    end_time = time.time()
    # 模型评估
    train_time = end_time - start_time
    accuracy = test()
    record(train_loss, params.EPOCHES, params.TRAIN_BATCH_SIZE, accuracy, train_time)
    torch.save(cnn, "saved_models/cnn.pth")
    torch.save(cnn.state_dict(), 'saved_models/cnn.params')


