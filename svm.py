import numpy as np
import matplotlib.pyplot as plt
from sklearn import svm
import scipy.io as scio

# 加载数据
dataset1 = scio.loadmat('F:\course\machine learning\MLA2_data\MLA2_data1.mat')
X1 = dataset1["X"]
Y1 = dataset1["y"]
dataset2 = scio.loadmat('F:\course\machine learning\MLA2_data\MLA2_data2.mat')
X2 = dataset2["X"]
Y2 = dataset2["y"]

# 计算支持向量分类
svc1 = svm.SVC(kernel='linear', C=100).fit(X1, Y1)
svc2 = svm.SVC(kernel='rbf', gamma=50, C=100).fit(X2, Y2)

# 绘图
def plot_svc(x, y, svc):
    x_min, x_max = x[:, 0].min() - 1, x[:, 0].max() + 1
    y_min, y_max = x[:, 1].min() - 1, x[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.02), np.arange(y_min, y_max, 0.02))
    z = svc.predict(np.c_[xx.ravel(), yy.ravel()])
    z = z.reshape(xx.shape)
    plt.contourf(xx, yy, z, cmap=plt.cm.coolwarm, alpha=0.8)
    plt.scatter(x[:, 0], x[:, 1], c=y, cmap=plt.cm.coolwarm)
    plt.xlim(xx.min(), xx.max())
    plt.ylim(yy.min(), yy.max())
    plt.xticks(())
    plt.yticks(())
    plt.show()

def plot_svc1(x, y, svc):
    x_min, x_max = x[:, 0].min() - 1, x[:, 0].max() + 1
    y_min, y_max = x[:, 1].min() - 1, x[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.02), np.arange(y_min, y_max, 0.02))
    z = svc.predict(np.c_[xx.ravel(), yy.ravel()])
    z = z.reshape(xx.shape)
    plt.contour(xx, yy, z, colors='k', alpha=0.8)
    plt.scatter(x[:, 0], x[:, 1], c=y, cmap=plt.cm.coolwarm)
    plt.xlim(0, 1.1)
    plt.ylim(0, 1.1)
    plt.xticks(())
    plt.yticks(())
    plt.show()

plt.title("linear")
plot_svc(X1, Y1, svc1)
plt.title("Gaussian Kernel")
plot_svc1(X2, Y2, svc2)
