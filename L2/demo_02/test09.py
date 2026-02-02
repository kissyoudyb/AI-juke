import torch

x = torch.tensor(2.0,requires_grad=True)
y = 2*x+3
y.backward()
print(x.grad)