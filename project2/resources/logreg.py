import numpy as np
import matplotlib.pyplot as plt
from machine_learning import MachineLearning


class LogisticRegression(MachineLearning):
    """ Class for own logistic regression """
    def __init__(self, n_batches = 1, allow_early_stop = True):
        """ Initialization of instance
        inputs:
        n_batches        : Number of batches to use in stochastic gradient
                           descent
        allow_early_stop : Whether or not to allow the epoch loop to stop early
                           if the solution stops improving
                           default = True
        """
        self._n_batches = n_batches
        self._allow_early_stop = allow_early_stop


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


    def SGD(self, X, y, n_epochs, eta = 0.01):
        """ Stochastic gradient descent method for input X and targets y
        input:
        X        : train feature matrix
        y        : train targets
        n_epochs : num of epochs
        eta      : learning rate
        """

        beta = self._beta
        M = self._n_batches

        self.accuracy_history = np.zeros(n_epochs + 1)
        self.cost_history = np.zeros(n_epochs + 1)

        self.accuracy_history[0] = self.accuracy()
        self.cost_history[0] = self.cost(X, y, beta)

        if (self._Xtest is not None) and (self._ytest is not None):
            """ if test inputs and targets are provided, calculate the accuracy
            and cost value for these as well
            """
            benchmark = True
            self.accuracy_hist_test = np.zeros(n_epochs + 1)
            self.cost_hist_test = np.zeros(n_epochs + 1)

            self.accuracy_hist_test[0] = self.accuracy(self._Xtest,
                                                       y_t = self._ytest)
            self.cost_hist_test[0] = self.cost(self._Xtest,
                                               self._ytest,
                                               beta)
        else:
            benchmark = False

        break_counter = 0

        minibatch_size = int(len(X)/M)
        for i in range(1, n_epochs+1):
            for j in range(M):
                random_index = np.random.randint(M)
                ind_start = random_index*minibatch_size
                ind_end = ind_start + minibatch_size
                X_batch = X[ind_start:ind_end]
                y_batch = y[ind_start:ind_end]

                beta = self.gradient_descent_step(X_batch,
                                                  y_batch,
                                                  beta,
                                                  eta,
                                                  minibatch_size)

            self.accuracy_history[i] = self.accuracy()
            self.cost_history[i] = self.cost(X, y, beta)

            if benchmark:
                self.accuracy_hist_test[i] = self.accuracy(self._Xtest,
                                                           y_t = self._ytest)
                self.cost_hist_test[i] = self.cost(self._Xtest,
                                                   self._ytest,
                                                   beta)


            if self.cost_history[i - 1] > (self.cost_history[i - 2] - 1e-10):
                """ if cost does not improve in five epochs, stop the loop, but
                allows a small tolerance.
                """
                break_counter += 1
                if break_counter > 4 and self._allow_early_stop:
                    print("Loss did not improve in five epochs, stopping")
                    break
            else:
                self._beta = beta # only store beta if cost improved
                break_counter = 0

            #print("Epoch: {:d} | Accuracy vs. training data: {:1.4f} | Loss: {:1.4f}"
            #    .format(i, self.accuracy_history[i - 1], self.cost_history[i - 1]))

        if benchmark:
            return (self.accuracy_history,
                    self.cost_history,
                    self.accuracy_hist_test,
                    self.cost_hist_test)


    def cost(self, X, y_data, beta):
        """ Method for getting the cost/loss for logistic regression
        X      : predictors
        y_data : target data (probabilities)
        beta   : beta parameters of model

        """
        y_model = X @ beta
        p = self.sigmoid(y_model)

        C = -np.mean(y_data*np.log(p) + (1 - y_data)*np.log(1+1e-15 - p))
        return C


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


    def fit(self, X, y, n_epochs=10, eta=0.01, newbeta = True,
            Xtest = None, ytest = None):
        print("Fitting with {:d} epochs, learning rate = {:g}".format(n_epochs, eta))
        self._X = X # predictors
        self._y = y # targets
        self._Xtest = Xtest
        self._ytest = ytest

        if newbeta:
            self._beta = np.random.rand(X.shape[1], 1) # initialize random beta

        return self.SGD(self._X, self._y, n_epochs, eta)


    def predict(self, X, beta = None):
        """ Make the prediction using a given X and beta """
        if beta is None:
            beta = self._beta

        y_pred = np.dot(X, beta)
        prob = self.sigmoid(y_pred)
        prob[prob>=0.5] = 1
        prob[prob<0.5] = 0

        return prob
