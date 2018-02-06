import numpy as np
import netmodel
import netdata as dm
import matplotlib.pyplot as plt

plt.style.use('fivethirtyeight')


def train():

    # If a file was downloaded
    if len(data) != 0:
        # Form input and output data for a model
        X = data[:, :-1]
        y = data[:, -1].astype(int)

        # Separate the data into train and test datasets
        X_train, X_test, y_train, y_test = dm.cross_val(X, y, test_size=0.25)

        # Save the column with names
        names_test = X_test[:, 0]

        # Delete the columns with name
        X_train, X_test = dm.delete_names(X_train, X_test)

        # Array to matrix(HotEncoding)
        y_train = dm.encoding(y_train)

        epsilon = 0.01

        num_passes = 100

        num_input = X_train.shape[1]  # input layer dimension
        num_output = y_train.shape[1]  # output layer dimension

        # Initialize a model
        model = netmodel.Neural(num_input, num_output, neurons_hlayer=30)

        # Build a model with a 3-dimensional hidden layer
        model.train_model(X_train, y_train, num_passes, epsilon, reg_lambda=0.001, print_loss=False)

        # Predict result using the test data
        result = model.predict(X_test) + 1

        # Count difference between real and predicted. Print accuracy
        diff = np.subtract(result, y_test)
        print("\nAccuracy: %d" % (100 - np.count_nonzero(diff)/diff.size*100) + "%\n\n")

        # Write a file (Format: type, name)
        with open("result.txt", 'w') as outfile:
            for i in range(len(names_test)):
                outfile.write(str(result[i]) + "->" + str(names_test[i]) + "\n")
        return 100 - np.count_nonzero(diff)/diff.size*100


"""
    The method which splits the data into the equal train and test sets 10 times

    Parameters:
    x_train: Data set with features;
    y_train: The correct results;
"""


def ten_cross_val():
    cv_results = []

    # If a file was downloaded
    if len(data) != 0:
        # Form input and output data for a model
        X = data[:, :-1]
        y = data[:, -1].astype(int)
        for i in range(10):
            # Separate the data into train and test datasets
            X_train, X_test, y_train, y_test = dm.cross_val(X, y, test_size=0.3, k=i)

            # Delete the columns with name
            X_train, X_test = dm.delete_names(X_train, X_test)

            # Array to matrix(HotEncoding)
            y_train = dm.encoding(y_train)

            epsilon = 0.01

            num_passes = 100

            num_input = X_train.shape[1]  # input layer dimension
            num_output = y_train.shape[1]  # output layer dimension

            # Initialize a model
            model = netmodel.Neural(num_input, num_output, neurons_hlayer=30)

            # Build a model with a 3-dimensional hidden layer
            model.train_model(X_train, y_train, num_passes, epsilon, reg_lambda=0.001, print_loss=False)

            # Predict result using the test data
            result = model.predict(X_test) + 1

            # Count difference between real and predicted. Print accuracy
            diff = np.subtract(result, y_test)
            accuracy = 100 - np.count_nonzero(diff) / diff.size * 100
            print("\nAccuracy: %d" % accuracy + "%\n\n")

            # Write a file (Format: type, name)
            cv_results.append(accuracy)
    print(np.mean(cv_results))


data = dm.readdata('zoo.data.txt')
means = []
stds = []
ten_cross_val()

for num_iter in range(100, 1001, 100):
    results = np.array([train() for i in range(num_iter)])

    with open(str(num_iter) + "_Models.txt", 'w') as outfile:
        outfile.write(str(list(results)))
    print("Mean:", np.mean(results))
    means.append(np.mean(results))
    print("Standard Deviation:", np.std(results))
    stds.append(np.std(results))

    plt.hist(results)
    plt.xlabel('Accuracy')
    plt.ylabel('Frequency')
    plt.title(str(num_iter) + ' Models')
    plt.xlim(60, 110)
    plt.grid(True)
    plt.savefig(str(num_iter) + '_Models_hist')
    plt.close()

means = np.array(means)
stds = np.array(stds)

r = np.arange(100, 1001, 100)

ax1 = plt.subplot(211)
plt.plot(r, means, label="Average")
ax1.legend(loc=3)
ax1.grid(True)

# share x and y
ax2 = plt.subplot(212, sharex=ax1)
plt.plot(r, stds,  label="STD")

plt.legend(loc=1)
plt.grid(True)
plt.savefig('Final_plots')
plt.show()