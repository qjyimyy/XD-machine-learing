import numpy as np
import matplotlib.pyplot as plt


# 定义学习率和训练次数
learning_rate = 0.01
epoch_n = 10000


def plot_decision_boundary(pred_func, X):

    x_min, x_max = X[:, 0].min() - .5, X[:, 0].max() + .5
    y_min, y_max = X[:, 1].min() - .5, X[:, 1].max() + .5
    h = 0.01
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
    Z = pred_func(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    plt.contourf(xx, yy, Z, cmap=plt.cm.Spectral)
    cValue = ['r', 'y', 'b', 'r']
    plt.scatter(X[:, 0], X[:, 1], c=cValue, cmap=plt.cm.Spectral)
    plt.show()

# 定义输入输出
X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
Y = np.array([[0, 1, 1, 0]]).T

# 定义激活函数
def sigmoid(t):
    return 1 / (1 + np.exp(-t))


class MLP(object):
    def __init__(self, input_size, hidden_size, output_size):
        '''
        初始化参数
        :param input_size:输入层大小
        :param hidden_size:隐藏层大小
        :param output_size:输出层大小
        '''
        self.w1 = np.random.randn(input_size, hidden_size) * np.sqrt(1/2)
        self.b1 = np.zeros(shape=(1, hidden_size))
        self.w2  = np.random.randn(hidden_size, output_size) * np.sqrt(1/2)
        self.b2 = np.zeros(shape=(1, output_size))
    def forward(self, x):
        '''
        定义前向传播
        '''
        self.L1 = np.dot(x, self.w1) + self.b1
        self.A1 = np.tanh(self.L1)
        self.L2 = np.dot(self.A1, self.w2) + self.b2
        self.A2 = sigmoid(self.L2)
        return self.A2

    def cost(self, y_pred, y):
        '''
        使用交叉熵损失函数
        :param y_pred: y的预测值
        :param y: 样本输出
        :return: 损失
        '''
        m = y.shape[0]  # 样本的数目
        cost = np.multiply(np.log(y_pred), y) + np.multiply((1 - y), np.log(1 - y_pred))
        cost = - np.sum(cost)/m
        cost = np.squeeze(cost)  # 保持维度
        return cost

    def backward(self, x, y):
        '''
        定义后向传播
        :param x: 输入
        :param y:样本输出
        '''
        m = x.shape[0]
        self.dL2 = self.A2 - y

        self.dw2 = (1. / m) * np.dot(self.A1.T, self.dL2)
        self.db2 = (1. / m) * np.sum(self.dL2, axis=0, keepdims=True)

        self.dL1 = np.multiply(np.dot(self.dL2, self.w2.T), 1-np.power(self.A1, 2))
        self.dw1 = (1. / m) * np.dot(x.T, self.dL1)
        self.db1 = (1. / m) * np.sum(self.dL1, axis=0, keepdims=True)
        # 更新参数
        self.w1 -= learning_rate * self.dw1
        self.b1 -= learning_rate * self.db1
        self.w2 -= learning_rate * self.dw2
        self.b2 -= learning_rate * self.db2

    def predict(self, x):
        '''
        生成预测
        :param x:输入
        :return:预测的标签
        '''
        return np.around(self.forward(x))

model = MLP(input_size=2, hidden_size=2, output_size=1)
for epoch in range(epoch_n):
    predictions = model.forward(X)
    cost = model.cost(predictions, Y)
    model.backward(X, Y)
    if epoch % 100 == 0:
        print("Epoch : {},  cost : {}".format(epoch, cost))
        print(model.predict(X).T)
        print(model.w1)

plot_decision_boundary(lambda x:model.forward(x), X)


