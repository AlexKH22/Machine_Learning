from sklearn.ensemble import RandomForestRegressor
from sklearn.cross_validation import KFold
from sklearn.cross_validation import cross_val_score
from pandas import read_csv
import numpy as np


data = read_csv('abalone.csv')
data['Sex'] = data['Sex'].map(lambda x: 1 if x == 'M' else (-1 if x == 'F' else 0))

x=data[data.columns[0:-1]]
y=data[data.columns[-1]]

fold = KFold(y.size, n_folds=5, shuffle=True, random_state=1)
j = 0

for i in range(1,51):
    rf = RandomForestRegressor(n_estimators=i, random_state=1)
    rf.fit(x,y)
    score = np.mean(cross_val_score(rf, x, y, 'r2', fold))

    print('Forest size:', str(i).rjust(2), '  Score:', '{0:.3f}'.format(score), '  > 0.52:',
          '+' if score > 0.52 else '-')

    if j == 0 and score > 0.52:
        j = i

file = open('Minfor_0.52.txt', 'w')
print(j, file=file, sep='', end='')
file.close()