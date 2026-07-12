import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.datasets import make_blobs

X,_ = make_blobs(random_state=3)
# plt.scatter(X[:,0],X[:,1])
# plt.show()

d_list = []

for i in range(1,10):
    kmeans = KMeans(n_clusters=i,init='random',random_state=0)
    kmeans.fit(X)
    d_list.append(kmeans.inertia_)

plt.plot(range(1,10),d_list,marker='+')
plt.xlabel('Number of Cluster')
plt.ylabel('Distortion')
plt.show()