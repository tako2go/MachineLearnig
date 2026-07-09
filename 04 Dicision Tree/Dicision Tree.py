import pandas as pd
from pandas import DataFrame
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier

# data={
# 'EV':[2,3,1,2,2,1,4], #説明変数ExplanationValue
# 'PV':[0,1,0,1,0,0,0], #目的変数PurposeValue(0連続or1離散)
# 'KR':['重回帰','ロジ回帰','単回帰','ロジ回帰','重回帰','単回帰','重回帰'],#回帰分析の種類KindOfRegression(単回帰,重回帰,ロジスティック回帰)
# }

# df1 = DataFrame(data)
# # print(df1)

# X = df1.loc[:,['EV','PV']].values
# y = df1['KR'].values

# X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.3,random_state=0)
# dtree = DecisionTreeClassifier(random_state=0)#DecisionTreeClassifierは学習中に乱数を使うことがある
# dtree.fit(X_train,y_train)

# new0 = [[5,1]]
# print(dtree.predict(new0))

import matplotlib.pyplot as plt
from sklearn.datasets import make_moons

X,y = make_moons(noise=0.3,random_state=0)#Xは座標、yは所属グループ
# plt.scatter(X[:,0],X[:,1],s=30,c=y,cmap='coolwarm')#y(所属グループ)によってc(color)を変更
# plt.show()

df_moon = DataFrame(X,columns=['x座標','y座興'])
df_moon['ラベル'] = y
# print(df_moon.head(10))

X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.3,random_state=0)
dtree2 = DecisionTreeClassifier(max_depth=3,random_state=0)#max_depth 木構造の深さ　深すぎる(分類が細かすぎる)と過学習のもとになる
dtree2.fit(X_train,y_train)

# print('train_score:',dtree2.score(X_train,y_train))
# print('test_score:',dtree2.score(X_test,y_test))

import sklearn.tree

# 決定木を可視化
plt.figure(figsize=(10, 5))
sklearn.tree.plot_tree(dtree2, # モデルの名前 
                       class_names=['○', '△'], # クラス名
                       filled=True # 色付きで表示
                       );
plt.show()