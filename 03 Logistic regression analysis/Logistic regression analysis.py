import numpy as np
from pandas import DataFrame
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

data2={
'TimeOfStudy':[5,3,4,7,2,8,6,10,4,6,2,6,3,5,8,8,7,5,3],
'Club':[1,1,0,0,1,0,1,1,0,1,0,0,0,1,0,0,1,0,0],
'TimeOfSmaPho':[3,5,4,3,3,2,4,1,4,6,5,2,2,3,3,1,2,1,5],
'Pass/Fail':[1,0,0,1,0,1,0,1,1,0,1,0,0,0,0,1,1,1,0]
}

data_exam = DataFrame(data2)
# print(data_exam)

X = data_exam.loc[:,['TimeOfStudy','Club','TimeOfSmaPho']].values
y = data_exam['Pass/Fail'].values
X_train,X_test,y_train,y_test = train_test_split(X,y,
                                                 test_size=0.4,random_state=0)
model = LogisticRegression()
model.fit(X_train,y_train)
# print(model.score(X_test,y_test))

new2 = [[6,0,2]]
print(model.predict(new2))