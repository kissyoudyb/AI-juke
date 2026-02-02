#用CPU运算
import torch

batch_size = 64
dim_in = 1000
hidden_layer = 100
dim_out = 10

# 创建随机的Tensor作为输入和输出
x = torch.randn(batch_size, dim_in)
y = torch.randn(batch_size, dim_out)

# 使用nn包来定义网络。nn.Sequential是一个包含其它模块(Module)的模块
# 每个Linear模块使用线性函数来计算，它会内部创建需要的weight和bias
model = torch.nn.Sequential(
    torch.nn.Linear(dim_in, hidden_layer),
    torch.nn.ReLU(),
    torch.nn.Linear(hidden_layer, dim_out)
)

# 常见的损失函数在nn包里也有，不需要我们自己实现
loss_fn = torch.nn.MSELoss()

learning_rate = 1e-4
for t in range(500):
    # 前向计算：通过x来计算y。Module对象会重写__call__函数， 因此我们可以把它当成函数来调用
    y_pred = model(x)

    # 计算loss
    loss = loss_fn(y_pred, y)
    print(t, loss.item())

    # 梯度清空，调用Sequential对象的zero_grad后所有里面的变量都会清零梯度
    model.zero_grad()

    # 反向计算梯度。我们通过Module定义的变量都会计算梯度
    loss.backward()

    # 更新参数，所有的参数都在model.parameters()里
    with torch.no_grad():
        for param in model.parameters():
            param -= learning_rate * param.grad
