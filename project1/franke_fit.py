import sys
sys.path.insert(1, 'resources/')

from franke import FrankeFunction, plot_franke
from OLS import *
from resources import *
import numpy as np
import matplotlib.pyplot as plt

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
z_noise = (z + np.random.normal(scale = 0.1, size = (N, N)))

plot_franke(x_m, y_m, z_noise)
plt.title("Data")

MSE_list = []
bias_list = []
var_list = []

m_list = [2, 3, 4, 5]

print("  m |     MSE    |    R2   | ")
print("#############################")

for m in m_list:
    X = design_matrix(x_m, y_m, m)
    """ OLS takes the design matrix and the data and immediately provides the
    prediction by making a call to the OLS-instance with a Vandermonde matrix
    for a set of predictors.
    """
    reg_fit = OLS(X, z_noise)
    z_predict = reg_fit(X)

    #plot_franke(x_m, y_m, z_predict.reshape((N,N)))
    title = "m = " + str(m)
    #plt.title(title)

    MSE_list.append(MSE(z_noise, z_predict))
    bias_list.append(bias(z_noise, z_predict))
    var_list.append(variance(z_predict))

    print("{:3d} | {:2.8f} | {:1.5f} |".format(m, MSE(z_noise, z_predict), R2(z_noise, z_predict)))

plt.show()

plt.plot(m_list, MSE_list)
plt.plot(m_list, var_list)
plt.plot(m_list, bias_list)
plt.legend(["MSE", "Variance", "Bias"])
plt.xlabel("Complexity of model [m]")
plt.ylabel("value")
plt.show()
