from PIL import Image
import torch
import numpy as np


def infer(data):
    y = cnn.forward(data)
    result = y.argmax(dim=1)
    return result

if __name__ == '__main__':
    print("请输入图片路径：")
    image_path = input()    # todo 判别路径是否合规    校验图片格式
    # 转换图像size，要改成[1, 1, 28, 28]这种形式，而且是Tensor向量
    img = Image.open(image_path).convert('L').resize((28, 28))
    data = np.array(img).reshape((1, 1, 28, 28))
    data = torch.Tensor(data)
    # 加载模型
    cnn = torch.load('./models/cnn.pth')
    # 进行推理
    result = infer(data).item()
    print("图像类别是: ", result)