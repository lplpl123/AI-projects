import numpy as np
from PIL import Image

image = Image.open("C:/Users/75882/Desktop/play/图片/别人/1.jpg")
image_array = np.array(image)
print(image_array.shape)
