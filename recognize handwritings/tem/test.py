from PIL import Image
import numpy as np
import torch


img = Image.open('C:/Users/p30030010/Desktop/temp/kadilake.jpg').convert('L')
img = img.resize((28, 28))
data = np.array(img)
data = data.reshape((1, 1, 28, 28))
data = torch.Tensor(data)
print(data.shape)
print(type(data))
print(data)