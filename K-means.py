import numpy as np
import matplotlib.pyplot as plt

from sklearn.cluster import KMeans  # Kmeans

# 加载数据
dataA_X = np.loadtxt('F:\course\machine learning\MLA1_cluster_data_text\cluster_data_dataA_X.txt')
dataA_Y = np.loadtxt('F:\course\machine learning\MLA1_cluster_data_text\cluster_data_dataA_Y.txt')
dataB_X = np.loadtxt('F:\course\machine learning\MLA1_cluster_data_text\cluster_data_dataB_X.txt')
dataB_Y = np.loadtxt('F:\course\machine learning\MLA1_cluster_data_text\cluster_data_dataB_Y.txt')
dataC_X = np.loadtxt('F:\course\machine learning\MLA1_cluster_data_text\cluster_data_dataC_X.txt')
dataC_Y = np.loadtxt('F:\course\machine learning\MLA1_cluster_data_text\cluster_data_dataC_Y.txt')

# 计算k-means中心以及标签
def kMeans(data):
    clf = KMeans(n_clusters=4)
    clf.fit(data.T)
    centers = clf.cluster_centers_  # 两组数据点的中心点
    labels = clf.labels_  # 每个数据点所属分组
    return centers, labels


centersA, labelsA = kMeans(dataA_X)
centersB, labelsB = kMeans(dataB_X)
centersC, labelsC = kMeans(dataC_X)
# 画出数据点
def plot_clf(data, labels):
    for i in range(len(labels)):
        if labels[i] == 0:
            plt.scatter(data[0][i], data[1][i], c='r')
        elif labels[i] == 1:
            plt.scatter(data[0][i], data[1][i], c='b')
        elif labels[i] == 2:
            plt.scatter(data[0][i], data[1][i], c='y')
        else:
            plt.scatter(data[0][i], data[1][i], c='g')

# 画出中心点
def plot_centers(centers):
    for i in range(4):
        plt.scatter(centers[i][0], centers[i][1], marker='*', s=100, c='k')
plot_clf(dataA_X, labelsA)
plot_centers(centersA)
plt.show()
plot_clf(dataB_X, labelsB)
plot_centers(centersB)
plt.show()
plot_clf(dataC_X, labelsC)
plot_centers(centersC)
plt.show()
