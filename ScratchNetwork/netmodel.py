import numpy as np
import netdata as dm


class Neural(object):
    """
        Neural Network with 3 layers

        Attributes:
        W1: The weights of the hidden layer;
        b1: The bias of the hidden layer;
        W2: The weights of the output layer;
        b2: The bias of the output layer;
        model: The dictionary which represents 3 layers Neural Network;
        iterations: Number of iterations during training;
        errors: Value of an error during the established iteration.
    """
    def __init__(self, num_input, num_output, neurons_hlayer):

        # Initialize the parameters to completely random values
        self.W1 = np.random.randn(num_input, neurons_hlayer)
        self.b1 = np.zeros((1, neurons_hlayer))
        self.W2 = np.random.randn(neurons_hlayer, num_output)
        self.b2 = np.zeros((1, num_output))

        # Initialize a dictionary for the parameters above
        self.model = {}

        self.m_t = [0]
        self.v_t = [0]
    """
        Static methods represent different activation functions
    """
    @staticmethod
    def sigmoid(x):
        return 1. / (1 + np.exp(-x))

    @staticmethod
    def dsigmoid(x):
        return x * (1. - x)

    @staticmethod
    def tanh(x):
        return np.tanh(x)

    @staticmethod
    def dtanh(x):
        return 1. - x * x

    """
        The method which trains the model

        Parameters:
        x_train: Data set with features;
        y_train: The correct results;
        num_passes: An amount of the iterations for the training;
        print_loss: if true then print the process of the training.

        Gradient descent parameters:
        reg_lambda: regularization strength;
        epsilon: learning rate for gradient descent.
    """
    def train_model(self, x_train, y_train, num_passes, epsilon, text_widget, reg_lambda=0.01, print_loss=False):
        # Check values
        if num_passes < 10:
            num_passes = 100
            text_widget.insert('end', 'Iterations were changed to optimal(100)\n\n', 'warning')
        if epsilon <= 0 or epsilon >= 0.1:
            epsilon = 0.01
            text_widget.insert('end', 'Learning rate was changed to optimal(0.01)\n\n', 'warning')

        for i in range(1, num_passes+1):
            # Forward propagation
            z1 = x_train.dot(self.W1) + self.b1
            a1 = self.tanh(z1)
            z2 = a1.dot(self.W2) + self.b2

            # Count the result
            probs = self.sigmoid(z2)

            # Back propagation in order to evaluate an error of the model and train it
            delta3 = probs - y_train
            dw2 = a1.T.dot(delta3)
            db2 = np.sum(delta3)

            # An error in the hidden layer
            delta2 = np.multiply(delta3.dot(self.W2.T), self.dtanh(a1))

            dw1 = x_train.T.dot(delta2)
            db1 = np.sum(delta2)

            # Regularization terms
            dw2 += reg_lambda * self.W2
            dw1 += reg_lambda * self.W1

            # An updating of the parameters
            self.W1 += -epsilon * dw1
            self.b1 += -epsilon * db1
            self.W2 += -epsilon * dw2
            self.b2 += -epsilon * db2

            # Print an log loss every 10 iterations
            if print_loss and i % 10 == 0:
                error = float(self.evaluate_loss(x_train, y_train, reg_lambda))
                error = "Log Loss during %i iteration: %.3f" % (i, error) + '\n'
                text_widget.insert('end', error, 'color')

                epsilon -= epsilon*0.1

    """
        The method which evaluates an error

        Parameters:
        x_train: Data set with features;
        y_train: The correct results;
        reg_lambda: regularization strength.
    """
    def evaluate_loss(self, x_train, y_train, reg_lambda):

        # Forward propagation
        z1 = x_train.dot(self.W1) + self.b1
        a1 = self.tanh(z1)
        z2 = a1.dot(self.W2) + self.b2

        # Count the result
        probs = self.sigmoid(z2)

        # Calculating the loss
        y_train = dm.decode(y_train)
        data_loss = -np.log(probs[range(len(x_train)), y_train])
        data_loss = np.sum(data_loss)

        # L2 regularization
        data_loss += reg_lambda * np.sum((np.sum(np.square(self.W1)), np.sum(np.square(self.W2))))

        return 1 / len(x_train) * data_loss

    """
        The method which predicts an output for a given data

        Parameters:
        x: Data set with features.
    """
    def predict(self, x):

        # Forward propagation
        z1 = x.dot(self.W1) + self.b1
        a1 = self.tanh(z1)
        z2 = a1.dot(self.W2) + self.b2

        # Count the result
        probs = self.sigmoid(z2)

        # Return a maximum probability in the array
        return np.argmax(probs, axis=1)

