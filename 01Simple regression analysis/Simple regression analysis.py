import pandas as pd
from pandas import DataFrame
import matplotlib.pyplot as plt
from sklearn import linear_model

data0={
'MaxTemp':[33,33,34,34,35,35,34,32,28,35,33,28,32,33,35,30,29,32,34,35],
'NumOfCus':[382,324,338,317,341,360,339,329,218,402,342,205,368,351,304,294,275,336,384,368]
}
Ice_Data = DataFrame(data0)
# print(Ice_Data)

# plt.scatter(Ice_Data['MaxTemp'],Ice_Data['NumOfCus'])#data名['列名'],data名['列名']
# plt.show()

model = linear_model.LinearRegression()#linear_modelというモジュールからLinearRegressionというクラスをインスタンス化

X = Ice_Data.loc[:,['MaxTemp']].values#.locは列名から取得　:を用いて前列から['MaxTemp]
Y = Ice_Data['NumOfCus'].values

model.fit(X,Y)
# print('RegCoef:',model.coef_)
# print('Inter:',model.intercept_)

# plt.scatter(Ice_Data['MaxTemp'],Ice_Data['NumOfCus'])
# plt.plot(X,model.predict(X),color='orange')
# plt.grid(True)
# plt.show()

# print('regressionScore:',model.score(X,Y))

new = [[31]]
print(model.predict(new))