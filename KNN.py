
from numpy import *
import operator as op
# 原始训练集
def createDateset():
    group = array([[1.0, 1.1], [1.0, 1.0], [0, 0], [0, 0.1]])
    labels = ['A', 'A', 'B', 'B']
    return group, labels

def classify0(intX, dataSet, labels, k):
    # intX是测试的用例，dataset训练集，labels是训练集对应的标签，k是用于选择最近邻的数目
    dataSetSize = dataSet.shape[0]
    # 用欧式距离公式进行距离计算
    # 计算A0-A1，B0-B1
    diffMat = tile(intX, (dataSetSize, 1)) - dataSet   # numpy.tile进行数组的重复生成
    sqdiffMat = diffMat**2
    # 求和开根
    sqDistances = sqdiffMat.sum(axis=1)
    distances = sqDistances**0.5

    sortedDistIndicies = distances.argsort()  # 返回的是数组值从小到大的索引值
    classCount = {}
    for i in range(k):
        voteIlabel = labels[sortedDistIndicies[i]]
        classCount[voteIlabel] = classCount.get(voteIlabel, 0) + 1
    sortedClassCount = sorted(classCount.items(), key=op.itemgetter(1), reverse=True)
    # python3中函数为：items(),python2中函数为：iteritems()
    return sortedClassCount[0][0]

group, labels = createDateset()
print(classify0([1.5, 0], group, labels, 3))
#  输入没有标签的新数据后，将新数据的每个特征与样本集中数据对应的特征进行比较
#  然后算法提取样本集中特征最相似数据的分类标签。一般来说取前K的
#  [1.5, 0]是新数据，group是样本集中的数据，3是k

