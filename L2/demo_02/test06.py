import torch

batch_size = 64
dim_in = 1000
hidden_layer = 100
dim_out = 10

x = torch.randn(batch_size, dim_in)
y = torch.randn(batch_size, dim_out)

model = torch.nn.Sequential(
	torch.nn.Linear(dim_in, hidden_layer),
	torch.nn.ReLU(),
	torch.nn.Linear(hidden_layer, dim_out),
)
loss_fn = torch.nn.MSELoss()

# 使用Adam算法，需要提供模型的参数和learning rate
optimizer = torch.optim.Adam(model.parameters())
for t in range(500):
	y_pred = model(x)

	loss = loss_fn(y_pred, y)
	print(t, loss.item())

	# 梯度清零，原来调用的是model.zero_grad，现在调用的是optimizer的zero_grad
	optimizer.zero_grad()

	loss.backward()

	# 调用optimizer.step实现参数更新
	optimizer.step()