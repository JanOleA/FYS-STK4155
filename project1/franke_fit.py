import sys
sys.path.insert(1, 'resources/')

from franke import FrankeFunction, plot_franke
from OLS import OLS
import numpy as np

def design_matrix(x, y, m):
    """ Sets up the design matrix for polynomials on the form
    [1, x, y, x², y², xy, x³, y³, x²y, xy², x⁴, y⁴, x³y, x²y², xy³, ...]
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

np.random.seed(1)

N = 1000
x = np.sort(np.random.uniform(0, 1, N))
y = np.sort(np.random.uniform(0, 1, N))
x_m, y_m = np.meshgrid(x, y)

z = FrankeFunction(x_m, y_m)
z_noise = (z + np.random.normal(scale = 1, size = (N, N)))

for m in [2, 3, 4, 5]:
    X = design_matrix(x_m, y_m, m)
    reg_fit = OLS(X, z_noise)
    z_predict = reg_fit(X)

    print("m:", m, "| MSE:", reg_fit.MSE(z), "| R2:", reg_fit.R2(z))
