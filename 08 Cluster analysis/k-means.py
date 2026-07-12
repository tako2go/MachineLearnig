import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs

X,_ = make_blobs(random_state=3)
# plt.scatter(X[:,0],X[:,1])
# plt.show()

df = pd.DataFrame(X)
# print(df.describe())

from sklearn.cluster import KMeans
kmeans = KMeans(init='random',n_clusters=3,random_state=42)
kmeans.fit(X)

y = kmeans.predict(X)#それぞれの要素がどのラベルに所属するか
dfy = pd.DataFrame(y)
# print(dfy)

mdf = pd.concat([df,dfy],axis=1)
mdf.columns = ['feature1','feature2','cluster']
# print(mdf)

#mdf.groupbyは指定した列の値ごとにデータをグループ化する
# for i,data in mdf.groupby('cluster'):
#     print("グループキー i =",i)
#     print(data)

for cluster_id in sorted(set(y)):
    cluster_data = mdf[y == cluster_id]
    plt.scatter(cluster_data.iloc[:,0],cluster_data.iloc[:,1],
                label=f'cluster={cluster_id}')
plt.title('Cluster Result')
plt.xlabel('feature1')
plt.ylabel('feature2')
plt.show()

