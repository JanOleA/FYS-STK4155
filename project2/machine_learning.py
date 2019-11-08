import numpy as np
import matplotlib.pyplot as plt


class MachineLearning:
    """ Parent class for common methods """
    def __init__(self):
        """ Initialization """

    def sigmoid(self, x):
        """ Sigmoid activation function """
        return np.exp(x)/(1 + np.exp(x))

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

    def softmax(self, x):
        """ Softmax activation function """
        return np.exp(x)/np.sum(np.exp(x), axis = 1, keepdims = True)

    def softmax_diff(self, x):
        """ Derivative of the softmax activation """
        return self.softmax(x)*(1 - self.softmax(x))

    def gradient_descent_step(self, X, y, beta, eta, N):
        """ Gradient descent method for a single step for one batch in
        stochastic gradient descent
        """
        y_model = X @ beta # prediction




class LogisticRegression(MachineLearning):
    """ Class for own logistic regression """
    def __init__(self, l = 0):
        self._l = l # optional regularization parameter

    def cost(y_data, y_model, l = None):
        """ Function for getting the cost/loss for logistic regression
        y_data : target data (probabilities)
        y_model : predicted data (probabilities)
        """
        if l is None:
            l = self._l

        C = -np.mean(y_data*np.log(y_model) + (1 - y_data)*np.log(1 - y_model))
        regu = self._l # wrong
        C += regu

        return C
