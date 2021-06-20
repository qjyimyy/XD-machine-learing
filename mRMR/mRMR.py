import mifs
import numpy as np

# 生成数据
def generateData(d, n):
    means = [1/(i**(1/2)) for i in range(1, d+1)]
    m1 = []
    m2 = []
    for mean in means:
        m1.append(np.random.normal(loc=mean, size=n))
    for mean in means:
        m2.append(np.random.normal(loc=-mean, size=n))
    return np.hstack((np.array(m1), np.array(m2)))


data = generateData(100, 10).T
y = np.ones((20), dtype='int64')
y[:10] = y[:10] * 0
# 调用mifs，使用MRMR算法
feat_selector = mifs.MutualInformationFeatureSelector(k=5, method='MRMR', n_features=10, verbose=1)
new_data = feat_selector.fit_transform(data, y)
