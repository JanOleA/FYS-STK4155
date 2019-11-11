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


    def softmax(self, x):
        """ Softmax activation function """
        return np.exp(x)/np.sum(np.exp(x), axis = 1, keepdims = True)


    def softmax_diff(self, x):
        """ Derivative of the softmax activation """
        return self.softmax(x)*(1 - self.softmax(x))


    def gradient_descent_step(self, X, y, beta, eta, N):
        """ Gradient descent method for a single step for one batch in
        stochastic gradient descent
        X    : feature matrix for current batch
        y    : targets for current batch
        beta : current guess
        eta  : learning rate
        N    : batch size

        returns:
        new beta values
        """

        y_model = np.dot(X, beta) # prediction
        p = self.sigmoid(y_model)

        dC_dbeta = -np.dot(X.T, (y - p))/N

        return beta - eta*dC_dbeta


    def accuracy_classification(self, y_target, y_model):
        """ Returns the accuracy score for classification problem """
        return np.sum(y_target == y_model)/len(y_target)
