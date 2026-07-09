import pandas as pd
from pandas import DataFrame
import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsClassifier#k-最近傍法(k-NN)
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score#正解率を計算するモジュール
from sklearn.datasets import make_moons 

X,y = make_moons(noise=0.3,random_state=0)
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.3,random_state=0)
kNN_model = KNeighborsClassifier()
kNN_model.fit(X_train,y_train)

# print("train_score:",kNN_model.score(X_train,y_train))
# print("test_score:",kNN_model.score(X_test,y_test))

# new0 = [[0.2,0.2]]
# print(kNN_model.predict(new0))

# NumPyをnpという名前でインポート
import numpy as np

# クラス0(y=0)のデータを散布図で表示
# x軸: 1列目の特徴量, y軸: 2列目の特徴量
# 点の大きさ30, マーカーは○
plt.scatter(X[y==0][:, 0], X[y==0][:, 1], s=30, marker='o')

# クラス1(y=1)のデータを散布図で表示
# 点の大きさ30, マーカーは^
plt.scatter(X[y==1][:, 0], X[y==1][:, 1], s=30, marker='^')

# 描画範囲(x軸)の最小値・最大値
x0_min, x0_max = -1.5, 2.3

# 描画範囲(y軸)の最小値・最大値
x1_min, x1_max = -1.2, 1.7

# クラスごとの塗りつぶし色を指定
color0, color1 = 'tab:blue', 'tab:orange'

# x軸方向に200個の点を等間隔で作成
f0 = np.linspace(x0_min, x0_max, 200)

# y軸方向に200個の点を等間隔で作成
f1 = np.linspace(x1_min, x1_max, 200)

# 格子状の座標を作成
# f0, f1はどちらも(200, 200)の配列になる
f0, f1 = np.meshgrid(f0, f1)

# 格子状の全ての座標に対してKNNで予測を行う
# reshape(-1, 1)で縦ベクトルに変換し、
# hstackで(x座標, y座標)を横に結合して予測データを作成
# 最後に元の格子形状(200×200)へ戻す
pred = kNN_model.predict(
    np.hstack([f0.reshape(-1, 1), f1.reshape(-1, 1)])
).reshape(f0.shape)

# 決定境界(クラスが切り替わる境界)を描画
plt.contour(f0, f1, pred, levels=[0.5])

# クラスごとの領域を半透明(alpha=0.25)で塗りつぶす
plt.contourf(
    f0,
    f1,
    pred,
    levels=1,
    colors=[color0, color1],
    alpha=0.25
)

plt.show()