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

clf = KNeighborsClassifier()
clf.fit(X,y)
x_min, x_max = X[:,0].min() - 1, X[:,0].max() + 1
y_min, y_max = X[:,1].min() - 1, X[:,1].max() + 1
xx,yy = np.meshgrid(np.arange(x_min,x_max, .02),
                    np.arange(y_min,y_max, .02))
Z = clf.predict(np.c_[xx.ravel(),yy.ravel()])
Z = Z.reshape(xx.shape)

plt.pcolormesh(xx,yy,Z, cmap=plt.cm.get_cmap('Pastel1'))
plt.scatter(X[:,0],X[:,1],c=y, cmap=plt.cm.get_cmap('spring'),edgecolors='k')
plt.xlim(xx.min(),xx.max())
plt.ylim(yy.min(),yy.max())
plt.title("Classifier:KNN")
plt.show()

