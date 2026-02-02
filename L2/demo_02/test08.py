import torch
import torch.nn as nn
import torch.optim as optim

# 定义一个简单的线性回归模型
class LinearRegression(nn.Module):
    def __init__(self):
        super(LinearRegression, self).__init__()
        self.fc = nn.Linear(1, 1)  # 输入1维，输出1维

    def forward(self, x):
        return self.fc(x)

# 初始化模型、损失函数和优化器
model = LinearRegression()
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.01)

# 训练数据
x = torch.tensor([[1.0], [2.0], [3.0]])
y_true = torch.tensor([[3.0], [5.0], [7.0]])

# 训练循环
for epoch in range(1600):  # 增加训练次数
    # 前向传播
    y_pred = model(x)
    loss = criterion(y_pred, y_true)

    # 反向传播
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    if epoch % 100 == 0:
        print(f"Epoch: {epoch}, Loss: {loss.item()}")

# 预测结果
print(f"预测结果：{model(torch.tensor([[4.0]])).item():.2f}")  # 应接近9.0
