import numpy as np
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter


def FrankeFunction(x, y):
    """ Returns the value of the Franke function at given values x and y """
    term1 = 0.75*np.exp(-(0.25*(9*x-2)**2) - 0.25*((9*y-2)**2))
    term2 = 0.75*np.exp(-((9*x+1)**2)/49.0 - 0.1*(9*y+1))
    term3 = 0.5*np.exp(-(9*x-7)**2/4.0 - 0.25*((9*y-3)**2))
    term4 = -0.2*np.exp(-(9*x-4)**2 - (9*y-7)**2)
    return term1 + term2 + term3 + term4


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


def design_matrix_linreg(x, y, m):
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
    y_data :    array containing the target data
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
    y_data :    array containing the target data
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
