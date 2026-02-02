import torch

dtype = torch.float
device = torch.device("cpu")
# device = torch.device("cuda:0") # 如果有GPU可以注释掉这行

batch_size = 64
dim_in = 1000
hidden_layer = 100
dim_out = 10

# 创建随机的Tensor作为输入和输出
# 输入和输出需要的requires_grad=False(默认)，
# 因为我们不需要计算loss对它们的梯度。
x = torch.randn(batch_size, dim_in, device=device, dtype=dtype)
y = torch.randn(batch_size, dim_out, device=device, dtype=dtype)

# 创建weight的Tensor，需要设置requires_grad=True
w1 = torch.randn(dim_in, hidden_layer, device=device, dtype=dtype, requires_grad=True)
w2 = torch.randn(hidden_layer, dim_out, device=device, dtype=dtype, requires_grad=True)

learning_rate = 1e-6
for t in range(500):
	# Forward阶段: mm实现矩阵乘法，但是它不支持broadcasting。
	# 如果需要broadcasting，可以使用matmul
	# clamp本来的用途是把值clamp到指定的范围，这里实现ReLU。
	y_pred = x.mm(w1).clamp(min=0).mm(w2)

	# pow(2)实现平方计算。
	# loss.item()得到这个tensor的值。也可以直接打印loss，这会打印很多附加信息。
	loss = (y_pred - y).pow(2).sum()
	print(t, loss.item())

	# 使用autograd进行反向计算。它会计算loss对所有对它有影响的
	# requires_grad=True的Tensor的梯度。

	loss.backward()

	# 手动使用梯度下降更新参数。一定要把更新的代码放到torch.no_grad()里
	# 否则下面的更新也会计算梯度。后面我们会使用torch.optim.SGD，
	# 它会帮我们管理这些用于更新梯度的计算。

	with torch.no_grad():
		w1 -= learning_rate * w1.grad
		w2 -= learning_rate * w2.grad

		# 手动把梯度清零
		w1.grad.zero_()
		w2.grad.zero_()