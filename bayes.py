import re
import numpy as np

# 创建数据集
def loadDataSet():
    fr = open('bayesPosts.txt')
    postList = []
    for i in fr.readlines():
        postList.append(re.split(r'[\s\n]+', i))
    classVec = [0, 1, 0, 1, 0, 1]  # 1是侮辱性， 0 是正常
    return postList, classVec


# 创建一个包含在所有文档中出现的不会重复词的列表
def createVocabList(dataSet):
    vocabSet = set([])
    for dox in dataSet:
        vocabSet = vocabSet | set(dox)  # 做并集
    return list(vocabSet)

# 表示词汇表中的单词在输入文档中是否出现（1是出现，0是不出现）
def setOfWords(vocabList, inputList):
    returnVec = [0]*len(vocabList)
    for word in inputList:
        if word in vocabList:
            returnVec[vocabList.index(word)] = 1
        else: print("单词：%s未出现" % word)
    return returnVec

'''
post, classVec = loadDataSet()
print(createVocabList(post))
print(setOfWords(createVocabList(post), post[0]))
'''
# 朴素贝叶斯分类器训练函数
def trainBayes(trainMatrix, trainCatgory):  # 输入文档矩阵，每篇文档类别标签
    numTrainDocs = len(trainMatrix)  # 训练文档的个数
    numWords = len(trainMatrix[0])  # 词汇表的大小
    P1 = sum(trainCatgory)/float(numTrainDocs)  # 分类1，即侮辱性词汇的个数
    P0Num = np.ones(numWords)
    P1Num = np.ones(numWords)  # 初始化为1,一般sigma=1
    P0Denom = 2.0; P1Denom = 2.0  # 有两个类别
    for i in range(numTrainDocs):
        if trainCatgory[i] == 1:
            P1Num += trainMatrix[i]  # y=1条件下，统计某个单词出现的个数
            P1Denom += sum(trainMatrix[i])  # 累计y=1的所有单词数量
        else:
            P0Num += trainMatrix[i]  # y=0条件下，统计某个单词出现的个数
            P0Denom += sum(trainMatrix[i])  # 累计y=0的所有单词数量
    # 防止相乘下溢，＋log
    P1Vect = np.log(P1Num / float(P1Denom))  # p(w/y=1)
    P0Vect = np.log(P0Num / float(P0Denom))  # p(w/y=0)
    return P0Vect, P1Vect, P1

'''
# 将所有值转化为0/1向量
trainMat = []
myVocabList = createVocabList(post)
for i in post:
    trainMat.append(setOfWords(myVocabList, i))
print(trainMat)
P0V, P1V, P1 = trainBayes(trainMat, classVec)
print(P1)  # 测试类别1的概率
'''

# 分类函数
def classify(testVec, P0Vec, P1Vec, PClass1):
    P1 = sum(testVec * P1Vec) + np.log(PClass1)  # 因为是log,所以这里是求和以及+号操作
    # p(y=1/w) = p(w/y=1) * p(y=1),注意这里需要乘上，testVec,过滤掉那些为0的特征的概率
    P0 = sum(testVec * P0Vec) + np.log(1.0 - PClass1)


    if P1 > P0:
        return 1
    else:
        return 0



post, classVec = loadDataSet()
# 将所有值转化为0/1向量
trainMat = []
myVocabList = createVocabList(post)
for i in post:
    trainMat.append(setOfWords(myVocabList, i))
print(trainMat)
P0V, P1V, P1 = trainBayes(trainMat, classVec)
# 测试分类函数
test0 = ['love', 'my', 'dalmation']
testVec = setOfWords(myVocabList, test0)
print(classify(testVec, P0V, P1V, P1))
test1 = ['stupid', 'garbage']
testVec = setOfWords(myVocabList, test1)
print(classify(testVec, P0V, P1V, P1))
test2 = ['my', 'dog', 'stupid']
testVec = setOfWords(myVocabList, test2)
print(classify(testVec, P0V, P1V, P1))





