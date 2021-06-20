import numpy as np
import torch

# 定义学习率和训练次数
learning_rate = 0.01
epoch_n = 1000

# 定义输入输出
# x = [(0, 0, 0), (0, 0, 1), (0, 1, 0), (0, 1, 1), (1, 0, 0), (1, 0, 1), (1, 1, 0), (1, 1, 1)]
# y = [0, 1, 1, 0, 1, 0, 0, 1]
x = [(0, 0), (0, 1), (1, 0), (1, 1)]
y = [0, 1, 1, 0]
x = torch.from_numpy(np.asarray(x)).float()
y = torch.from_numpy(np.asarray(y).reshape((4, 1))).float()

# 定义损失函数(均方误差函数)
loss_func = torch.nn.MSELoss()


# 搭建模型 2-2-1
models = torch.nn.Sequential(
    torch.nn.Linear(2, 2),  # 从输入层到隐含层的线性变换
    torch.nn.Sigmoid(),
    torch.nn.Linear(2, 1)  # 从隐含层到输出层的线性变换
)

# 优化参数
optimzer = torch.optim.Adam(models.parameters(), lr=learning_rate)
for epoch in range(epoch_n):
    y_pred = models(x)  # 预期的y
    loss = loss_func(y_pred, y)  # 计算损失
    if epoch % 100 == 0:
        print("Epoch:{},Loss:{:.4f}".format(epoch, loss))
    optimzer.zero_grad()
    loss.backward()
    optimzer.step()  # 更新梯度

for i in x:
    print(1 if models(i).detach().numpy() > 0.5 else 0)


for i in models.state_dict().items():
    print(i)


