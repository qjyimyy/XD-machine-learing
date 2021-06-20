import numpy as np
import matplotlib.pyplot as plt
from sklearn import svm

# 生成数据
xx, yy = np.meshgrid(np.linspace(-3, 3, 500),
                     np.linspace(-3, 3, 500))
np.random.seed(1)  # 让每次产生的随机数都相同，方便比较
X = np.random.randn(300, 2)
Y = np.logical_xor(X[:, 0] > 0, X[:, 1] > 0)
# 支持向量分类
# 多项式阶数
degree = 2

clf = svm.SVC(kernel='poly', degree=degree, C=1.0)
# C为惩罚系数，越大容易过拟合，越小容易欠拟合
clf.fit(X, Y)

# 绘制决策函数
Z = clf.decision_function(np.c_[xx.ravel(), yy.ravel()])
Z = Z.reshape(xx.shape)

def plot(Z, xx, yy):
    plt.imshow(Z, interpolation='nearest',
               extent=(xx.min(), xx.max(), yy.min(), yy.max()),
               aspect='auto',
               origin='lower',
               cmap=plt.cm.PuOr_r)

    plt.contour(xx, yy, Z, levels=[0], linewidths=2)
    plt.scatter(X[:, 0], X[:, 1], s=30, c=Y, cmap=plt.cm.Paired, edgecolors='k')
    plt.xticks(())
    plt.yticks(())
    plt.axis([-3, 3, -3, 3])
    plt.title("SVC with polynomial kernel, degree = 2")


plot(Z, xx, yy)
print("输出正类和负类支持向量总个数:", clf.n_support_)
print("输出正类和负类支持向量索引:", clf.support_)
print("输出正类和负类支持向量:", "\n", clf.support_vectors_)
plt.show()