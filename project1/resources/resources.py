import numpy as np
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter


def plot_surface(x, y, z, zlim=(-0.10, 1.40), show = False):
    """ Plots a 2D surface """

    fig = plt.figure()
    ax = fig.gca(projection="3d")

    # Plot the surface.
    surf = ax.plot_surface(x, y, z, cmap=cm.coolwarm,
                           linewidth=0, antialiased=False)

    # Customize the z axis.
    ax.set_zlim(zlim[0], zlim[1])
    ax.zaxis.set_major_locator(LinearLocator(10))
    ax.zaxis.set_major_formatter(FormatStrFormatter("%.02f"))

    # Add a color bar which maps values to colors.
    fig.colorbar(surf, shrink=0.5, aspect=5)

    if show: plt.show()


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


def confidence_interval(beta, X, zvar):
    """ Calculates the 95% confidence interval of beta by:
    CI = beta_i ± 1.96*sqrt(v_i)*sqrt(z_variance)
    Where v_i is the i-th diagonal element of (X.T*X)^-1

    Returns:
    (beta_min, beta_max)
    """
    vsqrt = np.sqrt(np.diag(np.linalg.pinv(X.T @ X)))
    beta_min = beta - 1.96*vsqrt*np.sqrt(zvar)
    beta_max = beta + 1.96*vsqrt*np.sqrt(zvar)

    return beta_min, beta_max


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

    return np.mean((y_data - y_model)**2)


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

    y_avg = np.mean(y_data)
    return 1 - (np.sum((y_data - y_model)**2)/np.sum((y_data - y_avg)**2))


def variance(y_model):
    """ Function for calculating and returning the variance of model data

    Inputs:
    y_model :   array containing the prediction data

    Returns:
    var :       the variance of y_model
    """

    return np.mean(np.var(y_model))


def bias(y_data, y_model):
    """ Function for calculating and returning the bias for a set of data and
    a prediction

    Inputs:
    y_data :    array containing the real data
    y_model :   array containing the prediction data

    Returns:
    bias :      value of the bias
    """

    return np.mean((y_data - np.mean(y_model))**2)
