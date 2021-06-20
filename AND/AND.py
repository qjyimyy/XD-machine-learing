import numpy as np
import matplotlib.pyplot as plt
from numpy.random import multivariate_normal
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

# 显示数据
def show_data(Px, Nx):
    plt.plot(Px[:, 0], Px[:, 1], 'r*', label='Positive')
    plt.plot(Nx[:, 0], Nx[:, 1], 'k+', label='Negative')
    plt.legend()

# 划线
def show_line(w, b, x_lsp):
    w = w.reshape(-1)
    Lx = np.linspace(x_lsp[0], x_lsp[1])
    Ly = -(w[0]*Lx+b)/w[1]
    plt.plot(Lx, Ly)


# 创建数据，train_test_split用以划分训练集和测试集
def create_data(sigma):
    Px = multivariate_normal(mean=[1, 1], cov=sigma*np.eye(2), size=300)
    Nx = np.concatenate((np.random.multivariate_normal(mean=(0, 0), cov=sigma * np.identity(2), size=100),
                           np.random.multivariate_normal(mean=(0, 1), cov=sigma * np.identity(2), size=100),
                           np.random.multivariate_normal(mean=(1, 0), cov=sigma * np.identity(2), size=100)))
    positive = [[x[0], x[1], 1] for x in Px]
    negative = [[x[0], x[1], -1] for x in Nx]
    train, test = train_test_split(positive + negative, test_size=0.3)
    return train, test, Px, Nx

# 训练感知机
class perception:
    def __init__(self):
        self.w = np.zeros(2)  # 使用2维
        self.b = 1

    def output(self, input):
        return np.sign(np.matmul(input, self.w.T) + self.b)

    def train(self, train, times):
        for i in range(times):
            deltas = []
            for j in train:
                delta = j[len(self.w):] - self.output(j[:len(self.w)])
                deltas.append(delta)
                self.b += delta
                self.w += delta * j[:len(self.w)]
            if abs(sum(deltas)) < 10E-5:
                break

plt.figure(figsize=(8, 5))
sigma = 0.1
x, y, Px, Nx = create_data(sigma)
per = perception()
per.train(x, 1000)
x_lsp = [np.min(x, axis=0)[0], np.max(x, axis=0)[0]]
true = [t[2] for t in y]
test = [per.output(t[:2]) for t in y]
print(classification_report(true, test))  # 用于显示主要分类指标的文本报告．在报告中显示每个类的精确度，召回率，F1值等信息。
show_data(Px, Nx)
show_line(per.w, per.b, x_lsp)
plt.title('sigma = %.3f' % sigma)
plt.show()