import quandl, math
import numpy as np
import datetime
import matplotlib.pyplot as plt
import pickle
from sklearn import preprocessing, cross_validation, svm
from sklearn.linear_model import LinearRegression
from matplotlib import style

style.use('ggplot')

def otherclf():
    for k in ['linear', 'poly', 'rbf', 'sigmoid']:
        clf = svm.SVR(kernel=k)
        clf.fit(X_train, y_train)
        confidence = clf.score(X_test, y_test)
        print(k, confidence)

df = quandl.get("WIKI/GOOGL")

print(df.head())

df = df[['Adj. Open',  'Adj. High',  'Adj. Low',  'Adj. Close', 'Adj. Volume']]

df['HL_PCT'] = (df['Adj. High'] - df['Adj. Low']) / df['Adj. Close'] * 100.0
df['PCT_change'] = (df['Adj. Close'] - df['Adj. Open']) / df['Adj. Open'] * 100.0

df = df[['Adj. Close', 'HL_PCT', 'PCT_change', 'Adj. Volume']]
print(df.head())

forecast_col = 'Adj. Close'
df.fillna(value=-99999, inplace=True)
forecast_out = int(math.ceil(0.01 * len(df)))

df['label'] = df[forecast_col].shift(-forecast_out)

X = np.array(df.drop(['label'], 1))
X = preprocessing.scale(X)
X_lately = X[-forecast_out:]
X = X[:-forecast_out]

df.dropna(inplace=True)

y = np.array(df['label'])

X_train, X_test, y_train, y_test = cross_validation.train_test_split(X, y, test_size=0.2)

clf = LinearRegression()

clf.fit(X_train, y_train)
confidence = clf.score(X_test, y_test)
print("Accuracy(LinearRegression): ", confidence)

clf = svm.SVR()

clf.fit(X_train, y_train)
confidence = clf.score(X_test, y_test)
print("Accuracy(SVM): ", confidence)

otherclf()

# clf = LinearRegression(n_jobs=-1)   # training part will be significantly faster
# clf.fit(X_train, y_train)
#
# with open('linearregression.pickle', 'wb') as f:
#     pickle.dump(clf, f)

pickle_in = open('linearregression.pickle', 'rb')
clf = pickle.load(pickle_in)

forecast_set = clf.predict(X_lately)
print("Predicted price:\n", forecast_set, "\nAccuracy:", confidence, "\nAmount of days:", forecast_out)

df['Forecast'] = np.nan     # fill all with NAN

last_date = df.iloc[-1].name
last_unix = last_date.timestamp()
one_day = 86400     # seconds in day
next_unix = last_unix + one_day

for i in forecast_set:
    next_date = datetime.datetime.fromtimestamp(next_unix)
    next_unix += 86400
    df.loc[next_date] = [np.nan for _ in range(len(df.columns)-1)]+[i]

print(df.tail())

df['Adj. Close'].plot()
df['Forecast'].plot()
plt.legend(loc=4)
plt.xlabel('Date')
plt.ylabel('Price')
plt.show()

