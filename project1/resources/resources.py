import numpy as np


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

    d2 = (y_data - y_model)**2
    return 1./n * np.sum(d2)


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

    # could also do np.sum((y_model - np.mean(y_model))**2)/len(y_model)
    return np.mean(y_model**2) - np.mean(y_model)**2


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
