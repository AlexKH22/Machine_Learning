import numpy as np
import pandas
from sklearn.tree import DecisionTreeClassifier


data = pandas.read_csv('titanic.csv')

del data["Cabin"]
del data["Embarked"]
del data["Name"]
del data["Parch"]
del data["SibSp"]
del data["Ticket"]
del data["PassengerId"]


for i in range(0,len(data['Sex'])):
    if (data['Sex'][i]=='male'):
        data['Sex'][i] = 1
    else:
        data['Sex'][i] = 0

data = data.dropna(subset=['Age', 'Sex', 'Fare', 'Pclass'], how='any')

y = np.array(data['Survived'].tolist())

del data["Survived"]


X = []
for i in range(0,len(data)):
    X.append(np.array(data.iloc[i]))


clf = DecisionTreeClassifier(random_state=241)
clf.fit(X, y)

print(clf.feature_importances_)