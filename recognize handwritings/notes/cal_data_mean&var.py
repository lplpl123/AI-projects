import torch
import numpy as np
from torchvision import transforms

data = np.array([
    [[1, 1, 1], [1, 1, 1], [1, 1, 1], [1, 1, 1], [1, 1, 1]],
    [[2, 2, 2], [2, 2, 2], [2, 2, 2], [2, 2, 2], [2, 2, 2]],
    [[3, 3, 3], [3, 3, 3], [3, 3, 3], [3, 3, 3], [3, 3, 3]],
    [[4, 4, 4], [4, 4, 4], [4, 4, 4], [4, 4, 4], [4, 4, 4]],
    [[5, 5, 5], [5, 5, 5], [5, 5, 5], [5, 5, 5], [5, 5, 5]]
], dtype='uint8')

data = transforms.ToTensor(data)
data = torch.unsqueeze(data, 0)

nb_samples = 0.

channel_mean = torch.zeros(3)
channel_std = torch.zeros(3)
print(data.shape)
N, C, H, W = data.shape[:4]
data = data.view(N, C, -1)
print(data.shape)

channel_mean += data.mean(2).sum(0)
channel_std += data.std(2).sum(0)

nb_samples += N

channel_mean /= nb_samples
channel_std /= nb_samples
print(channel_mean, channel_std)