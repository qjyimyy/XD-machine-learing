from math import log
import operator
import matplotlib.pyplot as plt

# 创建数据集
def createDataSet():
    dataSet = [[1, 1, 'yes'], [1, 1, 'yes'], [1, 0, 'no'], [0, 1, 'no'], [0, 1, 'no']]
    labels = ['no surfacing', 'flippers']
    return dataSet, labels


# 计算熵值
def calEnt(dataSet):
    numEnt = len(dataSet)
    labelCounts = {}
    for i in dataSet:
        currentLabel = i[-1]  # 创建字典，key值为最后一列
        if currentLabel not in labelCounts.keys():  # 如果不存在，则扩展字典并将当前键值加入字典
            labelCounts[currentLabel] = 0
        labelCounts[currentLabel] += 1  # 字典每个键值记录当前类别出现的次数
    Ent = 0.0
    for key in labelCounts:
        porb = float(labelCounts[key])/numEnt
        Ent -= porb*log(porb, 2)  # 计算熵值
    return Ent

#  划分数据集
#  将符合特征的数据抽取出来
def splitDataSet(dataSet, axis, value):  # 待划分的数据集，划分的特征，需要进行比较的特征值
    retDataSet = []
    for i in dataSet:
        if i[axis] == value:
            reduced = i[:axis]
            reduced.extend(i[axis+1:])  # 在末位一次性添加另一个序列多个值
            retDataSet.append(reduced)
    return retDataSet

#  选择最好的数据集划分
def chooseBest(dataSet):
    numFet = len(dataSet[0]) - 1  # 减去一个类标签，即为特征数
    bestEnt = calEnt(dataSet)
    bestGain = 0.0
    bestFet = -1
    for i in range(numFet):
        featList = [ex[i] for ex in dataSet]
        unique = set(featList)  # 创建唯一特征列表
        newEnt = 0.0
        for value in unique:
            spilt = splitDataSet(dataSet, i, value)
            prob = len(spilt)/float(len(dataSet))  # 计算权重，求得的新熵为，加权熵和
            newEnt += prob*calEnt(spilt)
        Gain = bestEnt - newEnt
        if(Gain > bestGain):
            bestGain = Gain
            bestFet = i
    return bestFet  # 输出最好划分的特征序号
# 判断最佳叶子节点：多数表决
def majority(classList):
    classCount = {}
    for i in classList:
        if i not in classCount.keys():  classCount[i] == 0
        classCount[i] += 1
    sort = sorted(classCount.items(), key=operator.itemgetter(1), reverse=True)
    return sort[0][0]

# 创建树
def createTree(dataSet, labels, featLabels):
    classList = [ex[-1] for ex in dataSet]
    if classList.count(classList[0]) == len(classList):return classList[0]  # 只有一个类别时停止划分
    if len(dataSet[0]) == 1: return majority(classList)  # 遍历完所有特征时返回出现次数最多的类标签
    bestFet = chooseBest(dataSet)  # 选择最好的划分集
    bestLabel = labels[bestFet]
    featLabels.append(bestLabel)
    mytree = {bestLabel: {}}
    del(labels[bestFet])
    featValue = [ex[bestFet] for ex in dataSet]
    unique = set(featValue)
    for i in unique:
        #subLabels = labels[:]
        #print(subLabels)
        mytree[bestLabel][i] = createTree(splitDataSet(dataSet, bestFet, i), labels, featLabels)
    return mytree

#  预测
def predict(inputTree, featlabels, testVec):
    # 第一棵树的根节点名称
    firstStr = list(inputTree.keys())[0]

    # 第二棵树的字典
    secondDict = inputTree[firstStr]
    # 第一棵树名称的索引值
    featIndex = featlabels.index(firstStr)
    # 将测试数据中对应的值取出来
    key = testVec[featIndex]
    # 得到第二棵树key对应的值
    value_2 = secondDict[key]
    # 如果对应的值为字典则迭代
    if isinstance(value_2, dict):
        classLabel = predict(value_2, featlabels, testVec)
    else:
        classLabel = value_2
    return classLabel

'''
函数说明:获取决策树叶子结点的数目

Parameters:
    myTree - 决策树
Returns:
    numLeafs - 决策树的叶子结点的数目
'''
def getNumLeafs(myTree):
    numLeafs = 0
    firstStr = list(myTree.keys())[0]
    secondDict = myTree[firstStr]
    for key in secondDict.keys():
        if type(secondDict[key]).__name__=='dict':#测试该结点是否为字典，如果不是字典，代表此结点为叶子结点
            numLeafs += getNumLeafs(secondDict[key])
        else:   numLeafs +=1
    return numLeafs

'''
函数说明:获取决策树的层数

Parameters:
    myTree - 决策树
Returns:
    maxDepth - 决策树的层数
'''
def getTreeDepth(myTree):
    maxDepth = 0
    firstStr = list(myTree.keys())[0]
    secondDict = myTree[firstStr]
    for key in secondDict.keys():
        if type(secondDict[key]).__name__=='dict':#测试该结点是否为字典，如果不是字典，代表此结点为叶子结点
            thisDepth = 1 + getTreeDepth(secondDict[key])
        else:   thisDepth = 1
        if thisDepth > maxDepth: maxDepth = thisDepth
    return maxDepth

'''
函数说明:绘制结点

Parameters:
    nodeTxt - 结点名
    centerPt - 文本位置
    parentPt - 标注的箭头位置
    nodeType - 结点格式
Returns:
    无
'''
def plotNode(nodeTxt, centerPt, parentPt, nodeType):
    arrow_args = dict(arrowstyle="<-")                                            #定义箭头格式
    createPlot.ax1.annotate(nodeTxt, xy=parentPt,  xycoords='axes fraction',
             xytext=centerPt, textcoords='axes fraction',
             va="center", ha="center", bbox=nodeType, arrowprops=arrow_args)

'''
函数说明:标注有向边属性值

Parameters:
    cntrPt、parentPt - 用于计算标注位置
    txtString - 标注的内容
Returns:
    无
'''
def plotMidText(cntrPt, parentPt, txtString):
    xMid = (parentPt[0]-cntrPt[0])/2.0 + cntrPt[0]
    yMid = (parentPt[1]-cntrPt[1])/2.0 + cntrPt[1]
    createPlot.ax1.text(xMid, yMid, txtString, va="center", ha="center", rotation=30)

    '''
函数说明:绘制决策树

Parameters:
    myTree - 决策树(字典)
    parentPt - 标注的内容
    nodeTxt - 结点名
Returns:
    无
'''
def plotTree(myTree, parentPt, nodeTxt):
    decisionNode = dict(boxstyle="sawtooth", fc="0.8")                                        #设置结点格式
    leafNode = dict(boxstyle="round4", fc="0.8")                                            #设置叶结点格式
    numLeafs = getNumLeafs(myTree)
    depth = getTreeDepth(myTree)
    firstStr = list(myTree.keys())[0]
    cntrPt = (plotTree.xOff + (1.0 + float(numLeafs))/2.0/plotTree.totalW, plotTree.yOff)  #中心位置
    plotMidText(cntrPt, parentPt, nodeTxt)
    plotNode(firstStr, cntrPt, parentPt, decisionNode)
    secondDict = myTree[firstStr]
    plotTree.yOff = plotTree.yOff - 1.0/plotTree.totalD             #y偏移
    for key in secondDict.keys():
        if type(secondDict[key]).__name__=='dict':#测试该结点是否为字典，如果不是字典，代表此结点为叶子结点
            plotTree(secondDict[key],cntrPt,str(key))
        else:
            plotTree.xOff = plotTree.xOff + 1.0/plotTree.totalW
            #不是叶结点，递归调用继续绘制
            plotNode(secondDict[key], (plotTree.xOff, plotTree.yOff), cntrPt, leafNode)
            plotMidText((plotTree.xOff, plotTree.yOff), cntrPt, str(key))#如果是叶结点，绘制叶结点，并标注有向边属性值
    plotTree.yOff = plotTree.yOff + 1.0/plotTree.totalD
'''
函数说明:创建绘制面板

Parameters:
    inTree - 决策树(字典)
Returns:
    无
'''
def createPlot(inTree):
    fig = plt.figure(1, facecolor='white')
    fig.clf()                                                #清空fig
    axprops = dict(xticks=[], yticks=[])
    createPlot.ax1 = plt.subplot(111, frameon=False, **axprops) #去掉x、y轴
    # createPlot.ax1 = plt.subplot(111, frameon=False) #ticks for demo puropses
    plotTree.totalW = float(getNumLeafs(inTree))
    plotTree.totalD = float(getTreeDepth(inTree))
    plotTree.xOff = -0.5/plotTree.totalW
    plotTree.yOff = 1.0
    plotTree(inTree, (0.5, 1.0), '')
    plt.show()

dataSet, labels = createDataSet()

'''
print(calEnt(dataSet))
print(splitDataSet(dataSet, 1, 1))
print(chooseBest(dataSet))

print(mytree)
'''
# 创建featLabels用来，重新获取labels
featLabels = labels

mytree = createTree(dataSet, labels, featLabels)
print(mytree)
print(getNumLeafs(mytree))
print(getTreeDepth(mytree))
print(predict(mytree, featLabels, [1, 1]))
createPlot(mytree)
