import numpy as np
import matplotlib.pyplot as plt
from machine_learning import MachineLearning
import sys


class NN_layer(MachineLearning):
    """ Class for making layer objects """
    def __init__(self, inputs, neurons,
                 activation="sigmoid",
                 w_init="rand_samp"):

        """ Initialization of layer object
        inputs:
        neurons  : number of neurons in the layer
        inputs   : number of inputs to layer
        """
        self.activation_name = activation
        if activation == "sigmoid":
            self.act_function = self.sigmoid
            self.activation_diff = self.sigmoid_diff
        elif activation == "softmax":
            self.act_function = self.softmax
            self.activation_diff = self.softmax_diff

        self._neurons = neurons
        self._inputs = inputs

        if w_init == "rand_samp":
            self.weights = ((2/np.sqrt(inputs)) *
                             np.random.random_sample((inputs, neurons))
                             - (1/np.sqrt(inputs)))
        elif w_init == "random":
            self.weights = np.random.rand(inputs, neurons)

        self.bias = np.zeros(neurons) #+ 1e-10


    def feedforward(self, X):
        """ Do the feedforward step for the current layer """
        z = (X @ self.weights) + self.bias
        self.z = z
        self.a = self.act_function(z)

        return self.a


    def __str__(self):
        return "NN layer, neurons: {:d}, act_function: {:s}".format(self._neurons, self.activation_name)


class NeuralNetwork(MachineLearning):
    """ Class for own neural network """
    def __init__(self, X, y, layers, n_batches = 1):
        """ Initialize the network
        inputs:
        X         : feature matrix
        y         : target matrix
        layers    : sizes of input and hidden layers
        n_batches : number of batches for SGD
        """
        self._X = X
        self._y = y

        self.inputs = X.shape[0]
        self.features = X.shape[1]
        outputs = y.shape[0]

        self._n_batches = n_batches
        minibatch_size = int(X.shape[0]/M)

        self._layers = []

        # set up layers
        for i, layer in enumerate(layers):
            if i == 0:
                self.add_layer(self.features, layer)
            else:
                self.add_layer(prev_layer, layer)
            prev_layer = layer

        # add output layer
        self.add_layer(prev_layer, y.shape[0],
                       activation="softmax",
                       w_init="random")


    def add_layer(self, inputs, neurons,
                  activation="sigmoid",
                  w_init="rand_samp"):
        """ Make a layer object and add it to the network """
        layer = NN_layer(inputs, neurons, activation, w_init)
        self._layers.append(layer)


    def feedforward(self, X = None):
        """ Does the feedforward step for all the layers
        inputs:
        X : feature matrix, if None uses the feature matrix from initialization
        """
        if X is None:
            a = self._X
        else:
            a = X

        for layer in self._layers:
            a = layer.feedforward(a)


    def backpropagation(self, X, y, eta=0.01, lmbda=0):
        """ Does the backpropagation for one minibatch
        inputs:
        X     : feature matrix used for the feedforward
        y     : target matrix
        eta   : learning rate
        lmbda : regularization parameter
        """

        deltas = []

        # output layer
        output_layer = self._layers[-1]
        delta_L = (y - output_layer.a)

        deltas.append(delta_L)

        prev_layer = output_layer

        # loop backwards through all layers except the output layer to calculate errors
        for layer in self._layers[:-1][::-1]:
            delta_l = deltas[0] @ (prev_layer.weights).T*layer.a*(1-layer.a)
            deltas.insert(0, delta_l)
            prev_layer = layer

        if X is None:
            a_prev = self._X
        else:
            a_prev = X

        N = a_prev.shape[0] # minibatch size for stochastic gradient descent

        # loop through all layers and apply gradient descent
        for delta, layer in zip(deltas, self._layers):
            W_grad = a_prev.T @ delta
            b_grad = np.sum(delta, axis = 0)

            regularization = lmbda * layer.weights
            layer.weights = layer.weights - eta*(W_grad + regularization)/N
            layer.bias = layer.bias - eta*b_grad/N

            a_prev = layer.a


    def fit(self, X = None, y = None, n_epochs = 10, eta = 0.01, newbeta = True):
        """ Fitting using stochastic gradient descent method for input X and
        targets y

        input:
        X        : training feature matrix, if None uses X from init
        y        : training targets, if None uses y from init
        n_epochs : num of epochs
        eta      : learning rate
        """

        if X is None:
            X = self._X
        if y is None:
            y = self._y

        print("Fitting with {:d} epochs, learning rate = {:g}".format(n_epochs, eta))

        M = self._n_batches

        self.accuracy_history = np.zeros(n_epochs)
        self.cost_history = np.zeros(n_epochs)

        break_counter = 0

        minibatch_size = int(X.shape[0]/M)
        for i in range(1, n_epochs+1):
            for j in range(M):
                random_index = np.random.randint(M)
                ind_start = random_index*minibatch_size
                ind_end = ind_start + minibatch_size
                X_batch = X[ind_start:ind_end]
                y_batch = y[ind_start:ind_end]

                self.feedforward(X_batch)
                self.backpropagation(X_batch, y_batch, eta)


            self.accuracy_history[i - 1] = self.accuracy()
            self.cost_history[i - 1] = 0

            print("Epoch: {:d} | Accuracy vs. training data: {:1.4f} | Loss: {:1.4f}"
                  .format(i, self.accuracy_history[i - 1], self.cost_history[i - 1]))


    def predict(self, X):
        """ Run feedforward and return output from final layer """
        self.feedforward(X)
        return self._layers[-1].a


    def accuracy(self, X = None, y_t = None):
        """ Returns accuracy using calculated beta
        Inputs:
        X   :   predictors to use for accuracy calculation,
                if None use training predictors
        y_p :   targets to use, if None use training targets
        """
        if X is None:
            X = self._X
        if y_t is None:
            y_t = self._y

        y_pred = self.predict(X)

        return self.accuracy_classification(y_t, y_pred)
