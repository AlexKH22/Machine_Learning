import numpy as np
import warnings
from collections import Counter
import pandas as pd
import random


def k_nearest_neighbors(data, predict, k=3):
    if len(data) >= k:
        warnings.warn('K is set to a value less than total voting groups!')
    distances = []
    for group in data:
        for features in data[group]:
            euclidean_distance = np.linalg.norm(np.array(features)-np.array(predict))
            distances.append([euclidean_distance,group])

    # print(distances)
    # print(sorted(distances))
    # print(sorted(distances)[:k])
    votes = [i[1] for i in sorted(distances)[:k]]
    # print(Counter(votes).most_common(1))  # list of tuples
    vote_result = Counter(votes).most_common(1)[0][0]
    confidence = Counter(votes).most_common(1)[0][1] / k

    return vote_result, confidence

df = pd.read_csv('breast-cancer-wisconsin.data.txt')
df.replace('?', -99999, inplace=True)
df.drop(['id'], 1, inplace=True)
full_data = df.astype(float).values.tolist()

accuracies = []
for i in range(10):
    random.shuffle(full_data)

    test_size = 0.2
    train_set = {2: [], 4: []}
    test_set = {2: [], 4: []}
    train_data = full_data[:-int(test_size*len(full_data))]
    test_data = full_data[-int(test_size*len(full_data)):]

    [train_set[i[-1]].append(i[:-1]) for i in train_data]

    [test_set[i[-1]].append(i[:-1]) for i in test_data]

    correct = 0
    total = 0

    for group in test_set:
        for data in test_set[group]:
            vote, confidence = k_nearest_neighbors(train_set, data, k=5)
            if group == vote:
                correct += 1
            else:
                print("Wrong result:", vote, "\nConfidence on the result:", confidence, "\n")
            total += 1
    print("Number of incorrect predictions:", total - correct)
    print('Accuracy:', correct/total)

    accuracies.append(correct/total)
print(2*"\n" + "#"*40, "\n" + "Average accuracy:", sum(accuracies)/len(accuracies))
