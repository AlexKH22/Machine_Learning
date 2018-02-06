from sklearn import datasets
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.grid_search import GridSearchCV
from sklearn.cross_validation import KFold
from sklearn.svm import SVC
import numpy as np

newsgroups = datasets.fetch_20newsgroups(
                    subset='all',
                    categories=['alt.atheism', 'sci.space']
             )

X = newsgroups.data
y = newsgroups.target

tdm = TfidfVectorizer()
XX = tdm.fit_transform(X)


grid = {'C': np.power(10.0, np.arange(-5, 6))}
cv = KFold(y.size,
           n_folds = 5,
           shuffle = True,
           random_state = 241)
clf = SVC(kernel = 'linear',
          random_state = 241)
gs = GridSearchCV(clf,
                  grid,
                  scoring = 'roc_auc',
                  cv = cv)
gs.fit(XX, y)

for a in gs.grid_scores_:
    print(a.mean_validation_score)
    print(a.parameters)
    # a.mean_validation_score — оценка качества по кросс-валидации
    # a.parameters — значения параметров
print(gs.best_params_)

print(np.argsort(abs(gs.best_estimator_.coef_.data)[::-1]))
temp = np.argsort(abs(gs.best_estimator_.coef_.data))
print(temp[-10:])
print(abs(gs.best_estimator_.coef_.data)[temp])

features = tdm.get_feature_names()
# print features
'''
temp = []
for word in words:
    temp.append(features[gs.best_estimator_.coef_.indices[word]])
temp.sort()
for x in temp:
    print x
print temp'''