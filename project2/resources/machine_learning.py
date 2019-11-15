import numpy as np
import matplotlib.pyplot as plt
import sys


class MachineLearning:
    """ Parent class for common methods """
    def __init__(self):
        """ Initialization """


    def sigmoid(self, x):
        """ Sigmoid activation function """
        return 1/(1 + np.exp(-x))


    def sigmoid_diff(self, x):
        """ Derivative of the sigmoid activation """
        return self.sigmoid(x)*(1 - self.sigmoid(x))


    def relu(self, x):
        """ ReLU activation function """
        return np.maximum(0, x)


    def relu_diff(self, x):
        """ Derivative of the ReLU activation """
        x[x <= 0] = 0
        x[x > 0] = 1

        return x


    def linear(self, x):
        return x


    def linear_diff(self, x):
        return 1


    def softmax(self, x):
        """ Softmax activation function """
        return np.exp(x)/np.sum(np.exp(x), axis = 1, keepdims = True)


    def softmax_diff(self, x):
        """ Derivative of the softmax activation """
        return self.softmax(x)*(1 - self.softmax(x))


    def accuracy_classification(self, y_target, y_model):
        """ Returns the accuracy score for classification problem (logreg) """
        return np.sum(y_target == y_model)/len(y_target)


    def accuracy_onehot(self, y_target, y_model):
        """ Returns the accuracy score for classification problem where the
        targets and predictions are set up as a onehot vector
        """
        target_indices = np.argmax(y_target, axis = 1)
        guess_indices = np.argmax(y_model, axis = 1)
        return np.sum(target_indices == guess_indices)/len(target_indices)
