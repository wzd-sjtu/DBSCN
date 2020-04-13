#  这里是DBSCN算法的调库使用

import numpy as np  # 数据结构
import sklearn.cluster as skc  # 密度聚类
from sklearn import metrics   # 评估模型
import matplotlib.pyplot as plt  # 可视化绘图
from sklearn import datasets

X1, y1=datasets.make_circles(n_samples=5000, factor=.6,
                                           noise=.05)
X2, y2 = datasets.make_blobs(n_samples=1000, n_features=2, centers=[[1.2,1.2]], cluster_std=[[.1]],
                random_state=9)
X = np.concatenate((X1, X2))

plt.scatter(X[:,0],X[:,1])
plt.show()

#  这里的参数选择十分玄妙
db = skc.DBSCAN(eps=0.08, min_samples=10).fit(X)
#  DBSCAN聚类方法 还有参数，matric = ""距离计算方法
labels = db.labels_
#  和X同一个维度，labels对应索引序号的值 为她所在簇的序号。若簇编号为-1，表示为噪声

raito = len(labels[labels[:] == -1]) / len(labels)  #计算噪声点个数占总数的比例
print('噪声比:', format(raito, '.2%'))

n_clusters_ = len(set(labels)) - (1 if -1 in labels else 0)  # 获取分簇的数目


#  在这里是进行的是简单的循环
for i in range(n_clusters_):
    print('簇 ', i, '的所有样本:')
    one_cluster = X[labels == i]
    print(one_cluster)
    plt.plot(one_cluster[:,0],one_cluster[:,1],'o')

plt.show()

