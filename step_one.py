#导入数据集生成器
from sklearn.datasets import  make_blobs
#导入分类器
from sklearn.neighbors import KNeighborsClassifier
#导入画图
import matplotlib.pyplot as plt
import numpy as np

#数据集拆分工具
from sklearn.model_selection import train_test_split

data = make_blobs(n_samples=200,centers=2,random_state=8)

X,y = data



plt.scatter(X[:,0],X[:,1],c=y, cmap=plt.cm.spring,edgecolors='k')
plt.show()

