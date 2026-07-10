import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.svm import LinearSVC

X,y = make_classification(n_samples=100,n_features=2,n_redundant=0,random_state=42)
#n_samples:サンプル数　n_features:特徴量 n_redundant:不要な特徴量

# plt.scatter(X[:,0],X[:,1],s=40,c=y,cmap='bwr')
# plt.grid(True)
# plt.show()
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.3,random_state=0)

model = LinearSVC()
model.fit(X_train,y_train)

# print("train_score:",model.score(X_train,y_train))
# print("test_score:",model.score(X_test,y_test))

plt.scatter(X[:,0],X[:,1],s=40,c=y,cmap='bwr')
X0 = np.linspace(-2.8,2.8)
Y = -model.coef_[0][0]/model.coef_[0][1] * X0 - model.intercept_/model.coef_[0][1]
#model_coef:関数の係数が入っている　y = ax+bに戻す式

plt.plot(X0,Y)
plt.xlim(min(X[:,0])-0.4,max(X[:,0])+0.4)
plt.ylim(min(X[:,1])-0.4,max(X[:,1])+0.4)
plt.grid(True)
# plt.show()

print(model.coef_)