#让模型自动判断设备，优先选择GPU
import torch

# 定义设备，如果可用则使用GPU，否则使用CPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)
batchsize = 64
dimin = 1000
hiddenlayer = 100
dimout = 10

# 创建随机的Tensor作为输入和输出，并移动到设备上
x = torch.randn(batchsize, dimin, device=device)
y = torch.randn(batchsize, dimout, device=device)

# 使用nn包来定义网络。nn.Sequential是一个包含其它模块(Module)的模块
# 每个Linear模块使用线性函数来计算，它会内部创建需要的weight和bias
model = torch.nn.Sequential(
    torch.nn.Linear(dimin, hiddenlayer),
    torch.nn.ReLU(),
    torch.nn.Linear(hiddenlayer, dimout)
).to(device)  # 将模型移动到设备上

# 常见的损失函数在nn包里也有，不需要我们自己实现
lossfn = torch.nn.MSELoss()

learningrate = 1e-4

for t in range(500):
    # 前向计算：通过x来计算y。Module对象会重写call函数， 因此我们可以把它当成函数来调用
    ypred = model(x)

    # 计算loss
    loss = lossfn(ypred, y)
    print(t, loss.item())

    # 梯度清空，调用Sequential对象的zero_grad后所有里面的变量都会清零梯度
    model.zero_grad()

    # 反向计算梯度。我们通过Module定义的变量都会计算梯度
    loss.backward()

    # 更新参数，所有的参数都在model.parameters()里
    with torch.no_grad():
        for param in model.parameters():
            param -= learningrate * param.grad
