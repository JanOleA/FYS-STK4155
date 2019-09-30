import numpy as np
import sys

class OLS:
    """ Class for a general OLS regression method """
    def __init__(self, X, y):
        """ Initializes the class and stores the design matrix X and target
        values y internally in the class

        Inputs:
        X :     design matrix (matrix [n x p])
        y :     target values (array)

        Runs the regression method on initialization
        """

        self._X = X
        self._y = y

        self.fit()


    def fit(self):
        """ Calculates the parameters of beta given a design matrix X and target
        values y
        """

        X = self._X
        y = self._y

        if len(y.shape) > 1:
            y = y.ravel()

        try:
            self._beta = np.linalg.inv(X.T.dot(X)).dot(X.T).dot(y)
        except Exception as e:
            print("Could not invert X.T.dot(X), error:", e)
            self._beta = None


    @property
    def beta(self):
        """ Property for extracting the beta values """
        return self._beta


    def predict(self, X):
        """ Predicts values of y_tilde given a matrix X and parameters beta.
        y_tilde is returned and also stored internally in the array.

        Inputs:
        X :         matrix containing the points to calculate y_tilde (matrix)

        Returns:
        y_tilde :   predictions of y with the given X and beta
        """
        self.y_tilde = X @ self._beta
        return self.y_tilde


    def __call__(self, X):
        """ Calls the predict method """
        return self.predict(X)


    def MSE(self, y = None):
        """ Calculate and return the MSE of the prediction and the original
        target values y

        Keyword arguments:
        y :     Override the target values to compare with if you wish to
                compare the prediction with a different set of data

        Returns:
        mse :   Mean squared error of the prediction vs. the target values
        """
        if y is None:
            y = self._y

        if len(y.shape) > 1:
            y = y.ravel()

        y_t = self.y_tilde
        if len(y_t.shape) > 1:
            y_t = y_t.ravel()

        n = len(y_t)

        d2 = (y - y_t)**2
        mse = 1/n * np.sum(d2)

        return mse

    def R2(self, y = None):
        """ Calculate and return the R2 score of the prediction and the original
        target values y

        Keyword arguments:
        y :     Override the target values to compare with if you wish to
                compare the prediction with a different set of data

        Returns:
        r2 :    R2 score of the prediction vs. the target values
        """
        if y is None:
            y = self._y

        if len(y.shape) > 1:
            y = y.ravel()

        y_t = self.y_tilde
        if len(y_t.shape) > 1:
            y_t = y_t.ravel()

        y_avg = np.average(y)
        r2 = 1 - (np.sum((y - y_t)**2)/np.sum((y - y_avg)**2))
        return r2
