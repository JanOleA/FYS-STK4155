import numpy as np


def design_matrix(x, y, m):
    """ Function for setting up the design matrix for polynomials on the form
    [1, x, y, x², y², xy, x³, y³, x²y, xy², x⁴, y⁴, x³y, x²y², xy³, ...]

    Inputs:
    x :     predictor, values of x
    y :     predictor, values of y
    m :     degree of the polynomial

    Returns:
    X :     The constructed design matrix
    """
    if len(x.shape) > 1:
        x = np.ravel(x)
        y = np.ravel(y)

    N = len(x)
    p = int((m+1)*(m+2)/2) # number of columns in the design matrix

    X = np.ones((N,p))

    for i in range(1, m+1):
        q = int((i)*(i+1)/2)
        for k in range(i + 1):
            X[:,q+k] = x**(i-k) * y**(k)

    return X


def MSE(y_data, y_model):
    """ Function for calculating and returning the mean square error between
    a set of data and a prediction

    Inputs:
    y_data :    array containing the real data
    y_model :   array containing the prediction data

    Returns:
    MSE :       value of the mean square error
    """
    if len(y_data.shape) > 1:
        y_data = y_data.ravel()

    if len(y_model.shape) > 1:
        y_model = y_model.ravel()

    n = len(y_model)

    return np.sum((y_data - y_model)**2)/n


def R2(y_data, y_model):
    """ Function for calculating and returning the R2 score for a set of data
    and a prediction

    Inputs:
    y_data :    array containing the real data
    y_model :   array containing the prediction data

    Returns:
    R2 :        value of the R2 score
    """
    if len(y_data.shape) > 1:
        y_data = y_data.ravel()

    if len(y_model.shape) > 1:
        y_model = y_model.ravel()

    n = len(y_model)

    y_avg = np.average(y_data)
    return 1 - (np.sum((y_data - y_model)**2)/np.sum((y_data - y_avg)**2))


def variance(y_model):
    """ Function for calculating and returning the variance of model data

    Inputs:
    y_model :   array containing the prediction data

    Returns:
    var :       the variance
    """
    if len(y_model.shape) > 1:
        y_model.ravel()

    return np.sum((y_model - np.mean(y_model))**2)/len(y_model)
    #return np.mean(y_model**2) - np.mean(y_model)**2


def bias(y_data, y_model):
    """ Function for calculating and returning the bias for a set of data and
    a prediction

    Inputs:
    y_data :    array containing the real data
    y_model :   array containing the prediction data

    Returns:
    bias :      value of the bias
    """
    if len(y_data.shape) > 1:
        y_data = y_data.ravel()

    if len(y_model.shape) > 1:
        y_model = y_model.ravel()

    return np.sum((y_data - np.mean(y_model))**2)/len(y_data)
