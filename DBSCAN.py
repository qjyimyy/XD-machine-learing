import pandas as pd
import numpy as np
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from sklearn.cluster import DBSCAN
from sklearn.metrics.cluster import normalized_mutual_info_score

# 加载数据
BasalCells = pd.read_csv(
    r'F:\course\machine learning\MLA3_data\scRNAseq_BasalCells.csv', index_col=0)
EndothelialCells = pd.read_csv(
    r'F:\course\machine learning\MLA3_data\scRNAseq_EndothelialCells.csv', index_col=0)
Fibroblasts = pd.read_csv(
    r'F:\course\machine learning\MLA3_data\scRNAseq_Fibroblasts.csv', index_col=0)
LuminalEpithelialCells = pd.read_csv(
    r'F:\course\machine learning\MLA3_data\scRNAseq_LuminalEpithelialCells.csv', index_col=0)
Macrophages = pd.read_csv(
    r'F:\course\machine learning\MLA3_data\scRNAseq_Macrophages.csv', index_col=0)

# 将所有样本组合一起
Cells = np.vstack((BasalCells.values.T, EndothelialCells.values.T,
                     Fibroblasts.values.T, LuminalEpithelialCells.values.T, Macrophages.values.T))

# 降维处理
def deCells(Cells):
    variance = ((Cells-Cells.mean(axis=0))**2).sum(axis=0)
    # 保留方差较大的前2000维特征
    sortVar = np.argsort(-variance)[:2000]
    sortCells = Cells[:, sortVar]
    pca = PCA(n_components=50).fit(sortCells)
    lowCells = pca.transform(sortCells)
    return lowCells

# 利用DBSCAN进行聚类
def dbSCAN(lowcells, eps, min_samples):
    # eps:点和点之间的间距, min_sample:可以算作核心点的高密度区域的最少点个数
    dbscan = DBSCAN(eps=eps, min_samples=min_samples).fit(lowcells)
    # 返回标准化互信息NMI
    return normalized_mutual_info_score(types, dbscan.labels_)


size5 = [len(list(BasalCells.columns)), len(list(EndothelialCells.columns)), len(list(
    Fibroblasts.columns)), len(list(LuminalEpithelialCells.columns)), len(list(Macrophages.columns))]

# 提取类别信息
types = []
for num, j in enumerate(size5):
    types += [num for i in range(j)]
types = np.array(types)

# 降维处理
lowcells = deCells(Cells)
print(lowcells, lowcells.shape)

# 画图
plt.scatter(lowcells[:, 0:1], lowcells[:, 1:2], c=types)
plt.title('cells')
plt.show()

print(dbSCAN(lowcells, 10, 10))
print(dbSCAN(lowcells, 10, 15))
print(dbSCAN(lowcells, 10, 20))
print(dbSCAN(lowcells, 17, 20))
print(dbSCAN(lowcells, 18, 20))
print(dbSCAN(lowcells, 20, 20))
print(dbSCAN(lowcells, 20, 25))
