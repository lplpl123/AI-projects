from torchvision import datasets, transforms
import numpy as np


transform = transforms.Compose([
        transforms.ToTensor(),
    ])
train_data = datasets.MNIST(root="C:/Users/p30030010/Desktop/my world/projects/AI-projects/recognize handwritings/data/MNIST",
                            train=True,
                            transform=transform,
                            download=False)
data = [0] * 60000
for i, val in enumerate(train_data):
    data[i] = np.array(val[0])
data = np.array(data)
data = data.reshape((60000, -1))
mean = data.sum() / (60000*28*28)
std = np.std(data)
print("mean: ", mean)
print("std", std)


