import torch

if __name__ == '__main__':
    a = torch.tensor([[1, 2, 3],
                      [4, 5, 6]])
    b = a.max(dim=1)[0]
    c = a.max(dim=1)[1]
    d = a.argmax(dim=1)
    print(d)