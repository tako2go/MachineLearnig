import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler

r = np.random.RandomState(1)

X = np.dot(r.rand(2,2),r.randn(2,200)).T

sc = StandardScaler()
Xs = sc.fit_transform(X)
# print(X)
# plt.scatter(Xs[:,0],Xs[:,1])
# plt.show()

from sklearn.decomposition import PCA
pca = PCA(n_components=2)
pca.fit(Xs)
# print(pca.components_)
# print(pca.explained_variance_)
# print(pca.explained_variance_ratio_)

import matplotlib.ticker as ticker
plt.gca().get_xaxis().set_major_locator(ticker.MaxNLocator(integer=True))
# plt.plot([0]+list(np.cumsum(pca.explained_variance_ratio_)), '-o')
# plt.show()

from sklearn.datasets import load_iris
iris = load_iris()

X = pd.DataFrame(iris.data,columns=iris.feature_names)
# print(X.describe())

# import seaborn as sns
# sns.pairplot(X)
# plt.show()

XP = pca.fit_transform(X)

df = pd.DataFrame(XP,columns=['pc1','pc2'])
print(XP)

plt.figure(figsize=(8,8))
y = pd.Series(iris.target, name = 'y')
plt.scatter(df.iloc[:,0],df.iloc[:,1],c=y)
plt.title('PCA Result')
plt.xlabel('pc1')
plt.ylabel('pc2')
# plt.show()