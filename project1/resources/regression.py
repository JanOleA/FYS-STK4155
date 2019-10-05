import numpy as np
from sklearn.linear_model import Lasso as scikit_lasso


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

        self._fit()


    def _fit(self):
        """ Calculates the parameters of beta given a design matrix X and target
        values y
        """

        X = self._X
        y = self._y

        if len(y.shape) > 1:
            y = y.ravel()

        try:
            self._beta = np.linalg.pinv(X.T.dot(X)).dot(X.T).dot(y)
        except Exception as e:
            print("Could not invert X.T.dot(X), error:", e)
            self._beta = None


    def fit(self, X, y):
        """ User callable method for recalculating the fit with new values for
        X and y
        """
        self._X = X
        self._y = y
        self._fit()


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


class Ridge(OLS):
    """ Subclass of the OLS class using the Ridge method for fitting """
    def __init__(self, X, y, l = 1.):
        """ Initializes the class and stores the design matrix X and target
        values y internally in the class

        Inputs:
        X :     design matrix (matrix [n x p])
        y :     target values (array)
        l :     lambda value for the ridge regression (number)

        Runs the regression method on initialization
        """
        self._X = X
        self._y = y
        self._l = l

        self._fit()


    def _fit(self):
        """ Calculates the parameters of beta given a design matrix X and target
        values y using Ridge regression
        """

        X = self._X
        y = self._y
        l = self._l

        if len(y.shape) > 1:
            y = y.ravel()

        I = np.identity(X.shape[1])

        try:
            self._beta = np.linalg.pinv(X.T.dot(X) + l * I).dot(X.T).dot(y)
        except Exception as e:
            print("Could not invert X.T.dot(X), error:", e)
            self._beta = None


    def fit(self, X, y, l):
        """ User callable method for recalculating the fit with new values for
        X, y and lambda
        """
        self._l = l
        super().fit(X, y)


class Lasso(OLS):
    """ Subclass of the OLS class using the Lasso method for fitting """
    def __init__(self, X, y, a = 1e-8, fit_intercept = False):
        """ Initializes the class and stores the design matrix X and target
        values y internally in the class

        Inputs:
        X :     design matrix (matrix [n x p])
        y :     target values (array)
        a :     alpha value for the lasso regression (number)

        Kwargs:
        fit_intercept : Includes the intercept in the fitting (boolean)

        Runs the regression method on initialization
        """
        self._X = X
        self._y = y

        self._model = scikit_lasso(alpha=a, fit_intercept=fit_intercept)
        self._fit()


    def _fit(self):
        """ Calculates the parameters of beta given a design matrix X and target
        values y using Lasso regression from scikit learn
        """

        X = self._X[:,1:] # exclude intercept
        y = self._y

        if len(y.shape) > 1:
            y = np.ravel(y)

        self._model.fit(X, y)

        self._beta = self._model.coef_
