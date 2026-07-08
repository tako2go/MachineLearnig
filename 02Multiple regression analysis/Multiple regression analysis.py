import numpy as np
from pandas import DataFrame
import matplotlib.pyplot as plt
from sklearn import linear_model
data1={
'StID':['2019017','2020003','2020012','2020015','2020021','2020024','2020028','2020032','2020036','2020040','2020043','2020049'],
'NumOfAb':[4,0,2,1,1,0,0,3,1,5,2,0],
'SemiEx':[47,92,78,65,88,82,98,58,64,35,89,84],
'FinEx':[74,93,65,72,84,88,95,65,73,55,78,92]
}

score_data = DataFrame(data1)
# print(score_data)

#欠席数と中間テストの関係
# plt.scatter(score_data['NumOfAb'],score_data['FinEx'])
# plt.grid(True)
# plt.show()

#中間テストと期末テストの関係
# plt.scatter(score_data['SemiEx'],score_data['FinEx'])
# plt.grid(True)
# plt.show()

model = linear_model.LinearRegression()
X1 = score_data.loc[:,['NumOfAb','SemiEx']].values
Y1 = score_data['FinEx'].values
model.fit(X1,Y1)

# print('RefCoef:',model.coef_)
# print('Inter',model.intercept_)
# print('regressionScore:',model.score(X1,Y1))

new1 = [[1,66]]#一回欠席中間66点の場合
print(model.predict(new1))