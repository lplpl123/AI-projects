from PIL import Image
import numpy as np


img = Image.open('C:/Users/p30030010/Desktop/temp/kadilake.jpg').convert('L')
img = img.resize((28, 28))
data = np.array(img)
data = data.reshape((1, 1, 28, 28))
print(data.shape)
print(data)