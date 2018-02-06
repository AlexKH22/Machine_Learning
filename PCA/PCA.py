from sklearn.decomposition import PCA
from pandas import read_csv
import numpy as np

data = read_csv('close-prices.csv')
data = data[[x for x in data.columns if x != 'date']]


clf = PCA(n_components=10)
clf.fit(data)

print("Varince: ")
print(clf.explained_variance_ratio_)

sum = 0
i = 0

while sum <= 0.9:
    sum += clf.explained_variance_ratio_[i]
    i += 1
print("For 90% variance " + str(i) + " features nedeed")

component_0 = list(map(lambda x: x[0], clf.fit_transform(data)))


dj = read_csv('djia_index.csv')['^DJI'].tolist()
c = np.corrcoef(component_0, dj)[0][1]
print("Pirson: " + str(c))

maxx = 0
j = 0

for x in range(0, len(clf.components_[0])):
    if maxx < clf.components_[0][x]:
        maxx = clf.components_[0][x]
        j = x

print('Max weight:', maxx)
print('Company:   ', data.columns[j])


file = open('Result01.txt', 'w')
print(i, file=file, sep='', end='')
file.close()

file = open('Result02.txt', 'w')
print('{0:.2f}'.format(c), file=file, sep='', end='')
file.close()

file = open('Result03.txt', 'w')
print(data.columns[j], file=file, sep='', end='')
file.close()