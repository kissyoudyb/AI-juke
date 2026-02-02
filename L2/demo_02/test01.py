import numpy as np

# batch size是批量大小；dim_in是输入大小
# hidden_layer是隐层的大小；dim_out是输出大小

batch_size = 64
dim_in = 1000
hidden_layer = 100
dim_out = 10

# 随机产生输入与输出
x = np.random.randn(batch_size, dim_in)
y = np.random.randn(batch_size, dim_out)

# 随机初始化参数
w1 = np.random.randn(dim_in, hidden_layer)
w2 = np.random.randn(hidden_layer, dim_out)

learning_rate = 1e-6
for t in range(500):
	# 前向计算y
	h = x.dot(w1)
	h_relu = np.maximum(h, 0)
	y_pred = h_relu.dot(w2)

	# 计算loss
	loss = np.square(y_pred - y).sum()
	print(t, loss)

	# 反向计算梯度
	grad_y_pred = 2.0 * (y_pred - y)
	grad_w2 = h_relu.T.dot(grad_y_pred)
	grad_h_relu = grad_y_pred.dot(w2.T)
	grad_h = grad_h_relu.copy()
	grad_h[h < 0] = 0
	grad_w1 = x.T.dot(grad_h)

	# 更新参数
	w1 -= learning_rate * grad_w1
	w2 -= learning_rate * grad_w2