import numpy as np
import matplotlib.pyplot as plt
from machine_learning import MachineLearning
from resources import MSE, R2
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
        elif activation.lower() == "relu":
            self.act_function = self.relu
            self.activation_diff = self.relu_diff
        elif activation.lower() == "linear":
            self.act_function = self.linear
            self.activation_diff = self.linear_diff

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
    def __init__(self, X, y, layers,
                 Xtest = None, ytest = None):
        """ Initialize the network
        inputs:
        X         : feature matrix
        y         : target matrix
        layers    : sizes of input and hidden layers

        Xtest     : test inputs for benchmarking
        ytest     : test targets for benchmarking
        """
        self._X = X
        self._y = y

        self._Xtest = Xtest
        self._ytest = ytest

        self.inputs = X.shape[0]
        self.features = X.shape[1]

        self._layers = []

        # set up hidden layers
        for i, layer in enumerate(layers):
            if i == 0:
                self.add_layer(self.features, layer)
            else:
                self.add_layer(prev_layer, layer)
            prev_layer = layer

        # add output layer
        self.add_output_layer(prev_layer, y.shape[1])


    def add_output_layer(self, inputs, neurons):
        """ Specific method for making the output layer for classification
        problems. Have separate method for this so this can be changed easily
        for regression problem.
        """
        self.add_layer(inputs, neurons,
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
        delta_L = (output_layer.a - y)

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


    def fit(self, X = None, y = None, n_epochs = 10,
            eta = 0.01, lmbda = 0, n_batches = 1,
            verbose = False):
        """ Fitting using stochastic gradient descent method for input X and
        targets y

        input:
        X        : training feature matrix, if None uses X from init
        y        : training targets, if None uses y from init
        n_epochs : num of epochs
        eta      : learning rate
        n_batches : number of batches for SGD
        verbose  : whether to print extra info while training
        """

        if X is None:
            X = self._X
        if y is None:
            y = self._y

        minibatch_size = int(X.shape[0]/n_batches)

        print("Fitting with {:d} epochs, learning rate = {:g}".format(n_epochs, eta))
        print("batch size = {:d}, regularization param = {:g}".format(minibatch_size, lmbda))
        print("Initial guess accuracy vs. training data:", self.accuracy())

        self.accuracy_history = np.zeros(n_epochs + 1)
        self.cost_history = np.zeros(n_epochs + 1)

        self.accuracy_history[0] = self.accuracy()
        self.cost_history[0] = self.cost(lmbda = lmbda)

        if (self._Xtest is not None) and (self._ytest is not None):
            """ if test inputs and targets are provided, calculate the accuracy
            and cost value for these as well
            """
            benchmark = True
            self.accuracy_hist_test = np.zeros(n_epochs + 1)
            self.cost_hist_test = np.zeros(n_epochs + 1)

            self.accuracy_hist_test[0] = self.accuracy(self._Xtest,
                                                       y_t = self._ytest)
            self.cost_hist_test[0] = self.cost(y_t = self._ytest,
                                               lmbda = lmbda)
        else:
            benchmark = False

        indices = np.arange(X.shape[0])
        for i in range(1, n_epochs+1):
            for j in range(n_batches):
                np.random.shuffle(indices)
                rand_indices = indices[:minibatch_size]
                X_batch = X[rand_indices]
                y_batch = y[rand_indices]

                self.feedforward(X_batch)
                self.backpropagation(X_batch, y_batch, eta, lmbda)

            """ Should potentially have implemented some functionality for
            storing the weights and biases that gives the smallest cost or
            highest accuracy (vs test?)
            """
            self.accuracy_history[i] = self.accuracy()
            self.cost_history[i] = self.cost(lmbda = lmbda)

            if benchmark:
                self.accuracy_hist_test[i] = self.accuracy(self._Xtest,
                                                           y_t = self._ytest)
                self.cost_hist_test[i] = self.cost(y_t = self._ytest,
                                                   lmbda = lmbda)

            if verbose:
                print("Epoch: {:d} | Accuracy vs. training data: {:1.4f} | Loss: {:1.4f}"
                    .format(i, self.accuracy_history[i - 1], self.cost_history[i - 1]))

        if benchmark:
            return (self.accuracy_history,
                    self.cost_history,
                    self.accuracy_hist_test,
                    self.cost_hist_test)


    def predict(self, X):
        """ Run feedforward and return output from final layer """
        self.feedforward(X)
        return self._layers[-1].a


    def cost(self, y_t = None, lmbda = 0):
        """ Method for getting the cost/loss for the neural network.
        Uses the result from the previous feedforward iteration for the model.
        Inputs:
        y_t   : target values, if None uses training targets.
        lmbda : regularization parameter
        """
        if y_t is None:
            y_t = self._y

        a = self._layers[-1].a # output layer output
        w = self._layers[-1].weights # output layer weights

        s = 0

        for i in range(y_t.shape[1]):
            s += np.sum(y_t[:,i] * np.log(a[:,i])
                        + (1 - y_t[:,i])*np.log(1 - a[:,i]))

        div_factor = 1/len(y_t)
        s *= div_factor

        C = -s - (lmbda * np.linalg.norm(w))*div_factor
        return C


    def accuracy(self, X = None, y_t = None):
        """ Does a prediction and returns the accuracy
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

        return self.accuracy_onehot(y_t, y_pred)


class NeuralNetworkLinear(NeuralNetwork):
    """ Class for own neural network for regression problems """
    def __init__(self, X, y, layers,
                 Xtest = None, ytest = None):
        """ Initialize is identical to classification NN
        inputs:
        X         : feature matrix
        y         : target matrix
        layers    : sizes of input and hidden layers

        Xtest     : test inputs for benchmarking
        ytest     : test targets for benchmarking
        """
        super().__init__(X, y, layers, Xtest, ytest)


    def add_output_layer(self, inputs, neurons):
        """ Specific method for making the output layer for regression
        problems. Linear activation so this can output any value.
        """
        self.add_layer(inputs, neurons,
                       activation="linear",
                       w_init="random")


    def add_layer(self, inputs, neurons,
                  activation="relu",
                  w_init="rand_samp"):
        """ Make a layer object and add it to the network. For regression using
        ReLU activation for layers
        """
        layer = NN_layer(inputs, neurons, activation, w_init)
        self._layers.append(layer)


    def backpropagation(self, X, y, eta=0.01, lmbda=0):
        """ Does the backpropagation for one minibatch, regression case
        inputs:
        X     : feature matrix used for the feedforward
        y     : target matrix
        eta   : learning rate
        lmbda : regularization parameter
        """

        deltas = []

        # output layer
        output_layer = self._layers[-1]
        delta_L = output_layer.activation_diff(output_layer.z)*(output_layer.a - y)

        deltas.append(delta_L)

        prev_layer = output_layer

        # loop backwards through all layers except the output layer to calculate errors
        for layer in self._layers[:-1][::-1]:
            derivative = layer.activation_diff(layer.z)
            delta_l = deltas[0] @ (prev_layer.weights).T*derivative
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


    def cost(self, y_t = None, lmbda = 0):
        """ Method for getting the cost/loss for the neural network.
        Uses the result from the previous feedforward iteration for the model.
        Inputs:
        y_t   : target values, if None uses training targets.
        """
        if y_t is None:
            y_t = self._y

        a = self._layers[-1].a # output layer output

        return MSE(y_t, a)


    def accuracy(self, X = None, y_t = None):
        """ Does a prediction and returns the accuracy
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

        return R2(y_t, self._layers[-1].a)
