import random
import torch


class DynamicNet(torch.nn.Module):
    """
    定义一个动态神经网络模块，输入层和输出层固定，中间层数量随机（0 - 3 个）且参数共享。

    参数:
    nIn (int): 输入特征的维度。
    nHidden_layer (int): 隐藏层的维度。
    nOut (int): 输出特征的维度。
    """

    def __init__(self, nIn, nHidden_layer, nOut):
        # 调用父类 torch.nn.Module 的构造函数
        super(DynamicNet, self).__init__()
        # 定义输入层的线性变换，输入特征维度为 nIn，输出特征维度为 nHidden_layer
        self.input_linear = torch.nn.Linear(nIn, nHidden_layer)
        # 定义中间层的线性变换，中间层参数共享，输入特征维度为 nHidden_layer，输出特征维度为 nHidden_layer
        self.middle_linear = torch.nn.Linear(nHidden_layer, nHidden_layer)
        # 定义输出层的线性变换
        self.output_linear = torch.nn.Linear(nHidden_layer, nOut)

    def forward(self, x):
        # 输入和输出层是固定的，但是中间层的个数是随机的(0,1,2)，并且中间层的参数是共享的。
        # 因为每次计算的计算图是动态(实时)构造的，所以我们可以使用普通的Python流程控制代码比如for循环来实现。
        # 另外一点就是一个Module可以多次使用，这样就可以实现参数共享。

        # 对输入进行线性变换，并通过ReLU激活函数
        h_relu = self.input_linear(x).clamp(min=0)  # 对输出结果使用 clamp 函数进行裁剪，确保输出值的最小值为 0，相当于应用了 ReLU 激活函数
        # 当随机数为 0 时，直接计算输出
        y_pred = self.output_linear(h_relu)

        # 随机决定中间层的数量，范围是 0 到 3
        for _ in range(random.randint(0, 3)):
            h_relu = self.middle_linear(h_relu).clamp(min=0)
            y_pred = self.output_linear(h_relu)  # 重新计算输出
        return y_pred


# 定义批量大小、输入特征维度、隐藏层维度和输出特征维度
batch_size = 64
dim_in = 1000
hidden_layer = 83
dim_out = 10

# 生成随机输入数据和标签
x = torch.randn(batch_size, dim_in)
y = torch.randn(batch_size, dim_out)

# 创建 DynamicNet 模型实例
model = DynamicNet(dim_in, hidden_layer, dim_out)

# 定义均方误差损失函数，不进行平均处理
criterion = torch.nn.MSELoss(size_average=False)
# 定义随机梯度下降优化器，设置学习率和动量
optimizer = torch.optim.SGD(model.parameters(), lr=1e-4, momentum=0.9)

# 训练模型 500 个 epoch
for t in range(500):
    y_pred = model(x)

    # 计算损失值
    loss = criterion(y_pred, y)
    print(t, loss.item())

    # 清空优化器中的梯度信息
    optimizer.zero_grad()

    # 反向传播，计算梯度
    loss.backward()

    # 更新模型参数
    optimizer.step()