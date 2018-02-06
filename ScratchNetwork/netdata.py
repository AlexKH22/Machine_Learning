import numpy as np

"""
    The method which decodes the feature into the training and test sets
    From an array to a matrix

    Parameters:
    features: The features for encoding.
"""


def encoding(features):
    new_features = np.zeros((len(features), max(features)))
    new_features[range(len(features)), features[range(len(features))] - 1] = 1

    return new_features

"""
    The method which decodes the feature into the training and test sets
    From a matrix to an array

    Parameters:
    features: The features for decoding.
"""


def decode(features):
    return np.array([np.argmax(arr) + 1 for arr in features]) - 1

"""
    The method which reads the data from txt

    Parameters:
    path: The path to the chosen file.
"""


def readdata(path):
    df = []
    with open(path, 'r', encoding='utf-8') as infile:
        for line in infile:
            df.append(line[:-1].split(','))
    return np.array(df)

"""
    The method which splits the data into the train and test sets

    Parameters:
    x_train: Data set with features;
    y_train: The correct results;
    test_size: An split proportion;
    k: K cross validation
"""


def cross_val(x_train, y_train, test_size=0.2, k=-1):
    # fix random seed (the best result when np.random.seed(7))
    # np.random.seed(0)
    x_test = []
    y_test = []
    maxlen = len(x_train)
    j = 0

    if k > -1:
        for i in range(0+k*10, (k+1)*10):
            try:
                x_test.append(x_train[i])
                y_test.append(y_train[i])
            except IndexError:
                return False
        x_train = np.delete(x_train, range(0+k*10, (k+1)*10), 0)
        y_train = np.delete(y_train, range(0+k*10, (k+1)*10), 0)
        return x_train, np.array(x_test), y_train, np.array(y_test)

    for i in range(0, int(test_size * maxlen)):
        num = np.random.randint(-1, maxlen - j)
        x_test.append(x_train[num])
        y_test.append(y_train[num])
        x_train = np.delete(x_train, num, 0)
        y_train = np.delete(y_train, num, 0)
        j += 1

    return x_train, np.array(x_test), y_train, np.array(y_test)

"""
    The method which deletes names of the animals from the datasets

    Parameters:
    All datasets.
"""


def delete_names(x_train, x_test):
    x_train = np.delete(x_train, 0, 1).astype(int)
    x_test = np.delete(x_test, 0, 1).astype(int)
    return x_train, x_test

"""
    The method which calculates the average

    Parameters:
    Array.
"""


def average(array):
    return sum(array)/len(array)

"""
    The method which calculates standard deviation

    Parameters:
    Array.
"""


def std(array, mean):
    return (sum((array - mean)**2)/len(array))**(1/2)
