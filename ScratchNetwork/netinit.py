import numpy as np
import netmodel
import netdata as dm

data = []


def data_creation(name):

    global data
    # Read the data
    data = dm.readdata(name)

"""
    The method which splits the data into the equal train and test sets 10 times

    Parameters:
    X, y: Input data;
    text_widget: For text printing;
    epsilon, num_passes: Network parameters
"""


def ten_cross_val(X, y, text_widget, epsilon, num_passes):
    cv_results = []

    for i in range(10):
        # Separate the data into train and test datasets
        X_train, X_test, y_train, y_test = dm.cross_val(X, y, test_size=0.3, k=i)

        # Delete the columns with name
        X_train, X_test = dm.delete_names(X_train, X_test)

        # Array to matrix(HotEncoding)
        y_train = dm.encoding(y_train)

        num_input = X_train.shape[1]  # input layer dimension
        num_output = y_train.shape[1]  # output layer dimension

        model = modeltrain(num_input, num_output, X_train, y_train,
                   num_passes, epsilon)

        # Predict result using the test data
        result = model.predict(X_test) + 1

        # Count difference between real and predicted. Print accuracy
        diff = np.subtract(result, y_test)
        accuracy = 100 - np.count_nonzero(diff) / diff.size * 100

        # Write a file (Format: type, name)
        cv_results.append(accuracy)
    return np.array(cv_results)


def train(entries, text_widget):
    global data

    # If a file was downloaded
    if len(data) != 0:
        # Print text in a widget
        text_widget.delete(1.0, 'end')
        text_widget.insert('end', 'Model Training\n', 'big')

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

        if len(entries['Learning Rate'].get()) == 0:
            text_widget.insert('end', 'Learning rate was changed to optimal(0.01)\n\n', 'warning')
            epsilon = 0.01
        else:
            epsilon = float(entries['Learning Rate'].get())
        if len(entries['Number of Iterations'].get()) == 0:
            text_widget.insert('end', 'Iterations were changed to optimal(100)\n\n', 'warning')
            num_passes = 100
        else:
            num_passes = int(round(float(entries['Number of Iterations'].get()), 0))

        num_input = X_train.shape[1]  # input layer dimension
        num_output = y_train.shape[1]  # output layer dimension

        model = modeltrain(num_input, num_output, X_train, y_train,
                   num_passes, epsilon, text_widget)

        # Predict result using the test data
        result = model.predict(X_test) + 1

        # Count difference between real and predicted. Print accuracy
        diff = np.subtract(result, y_test)
        text_widget.insert('end', "\nAccuracy: %d" % (100 - np.count_nonzero(diff)/diff.size*100) + "%\n\n", 'color')

        # Write a file (Format: type, name)
        with open("result.txt", 'w') as outfile:
            text_widget.insert('end', "Results are located in 'result.txt'", 'color')
            for i in range(len(names_test)):
                outfile.write(str(result[i]) + "->" + str(names_test[i]) + "\n")

        ten_results = ten_cross_val(X, y, text_widget, epsilon, num_passes)

        text_widget.insert('end', "\n\n10-fold Cross Validation\n", 'big')
        for index, result in enumerate(ten_results):
            text_widget.insert('end', str(index + 1) + "-split: " + str(result) + "\n", 'color')
        text_widget.insert('end', "Average accuracy: %d" % (dm.average(ten_results)) + "%\n", 'color')
        text_widget.insert('end', "Stardard Deviation: %d" % (dm.std(ten_results, dm.average(ten_results))) + "\n",
                           'color')

        # Calculate average and std
        text_widget.insert('end', "\nEvaluation(100 models)\n", 'big')
        accuracy = []
        for i in range(100):
            # Separate the data into train and test datasets
            X_train, X_test, y_train, y_test = dm.cross_val(X, y, test_size=0.25)

            # Delete the columns with name
            X_train, X_test = dm.delete_names(X_train, X_test)

            # Array to matrix(HotEncoding)
            y_train = dm.encoding(y_train)

            model = modeltrain(num_input, num_output, X_train, y_train,
                               num_passes, epsilon, None)

            # Predict result using the test data
            result = model.predict(X_test) + 1

            # Count difference between real and predicted. Print accuracy
            diff = np.subtract(result, y_test)
            accuracy.append(100 - np.count_nonzero(diff) / diff.size * 100)

        accuracy = np.array(accuracy)
        text_widget.insert('end', "Average accuracy: %d" % (dm.average(accuracy)) + "%\n", 'color')
        text_widget.insert('end', "Standard Deviation: %d" % (dm.std(accuracy, dm.average(accuracy))) + "\n", 'color')

    else:
        # Error message
        text_widget.delete(1.0, 'end')
        text_widget.insert('end', 'Firstly choose a file\nwith'
                           ' name: zoo.data.txt\n', 'big')

"""
    The method which trains model

    Parameters:
        num_input: Amount of input neurons
        num_output: Amount of output neurons
        x_train: Data set with features;
        y_train: The correct results;
        num_passes: An amount of the iterations for the training;

        Gradient descent parameters:
        epsilon: learning rate for gradient descent;

        text_widget: For text printing.
"""


def modeltrain(num_input, num_output, X_train, y_train, num_passes, epsilon, text_widget=None):
    # Initialize a model
    model = netmodel.Neural(num_input, num_output, neurons_hlayer=30)

    # Build a model with a 3-dimensional hidden layer
    if text_widget is not None:
        model.train_model(X_train, y_train, num_passes, epsilon, text_widget, reg_lambda=0.001, print_loss=True)
    else:
        model.train_model(X_train, y_train, num_passes, epsilon, text_widget, reg_lambda=0.001, print_loss=False)

    return model
