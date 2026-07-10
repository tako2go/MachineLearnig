import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.datasets import make_moons
from sklearn.svm import SVC

X,y = make_moons(noise=0.1, random_state=0)
# plt.scatter(X[:,0],X[:,1],s = 30,c = y,cmap='bwr')
# plt.show()

X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.3,random_state=0)

model2 = SVC()
model2.fit(X_train,y_train)

# print("train_score:",model2.score(X_train,y_train))
# print("test_score:",model2.score(X_test,y_test))

plt.scatter(X[:,0],X[:,1],s=40,c=y,cmap='bwr')

X0_min,X0_max = -1.4, 2.2
X1_min,X1_max = -0.7,1.2
f0 = np.linspace(X0_min,X0_max,200)
f1 = np.linspace(X1_min,X1_max,200)
f0,f1 = np.meshgrid(f0,f1)
#格子状に座標を生成
#f0には座標すべての点のx座標、f1にはすべての点のY座標が入っている
#この二つを重ねると座標が出てくる

pred = model2.predict(np.hstack([
    f0.reshape(-1,1),f1.reshape(-1,1)])).reshape(f0.shape)
plt.contour(f0,f1,pred,levels = [0.5])
plt.show()
print(f0)