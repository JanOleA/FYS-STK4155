""" Testing the fitting without train/test split up to 5th order complexity.
Plots the various fits.
"""

import sys
sys.path.insert(1, '../resources/')

from franke import FrankeFunction
from regression import *
from resources import *
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Ridge as sk_Ridge

np.random.seed(1)

N = 1000
x = np.sort(np.random.uniform(0, 1, N))
y = np.sort(np.random.uniform(0, 1, N))
x_m, y_m = np.meshgrid(x, y)

z = FrankeFunction(x_m, y_m)
z_noise = (z + np.random.normal(scale = 0.1, size = (N, N)))

plot_surface(x_m, y_m, z)
plt.title("Franke function")
plt.xlabel("x")
plt.ylabel("y")
plt.savefig("franke.pdf")

m_list = [2, 3, 4, 5]

z = z_noise

MSE_list = []
var_list = []
bias_list = []

l = 10

for m in m_list:
    X = design_matrix(x_m, y_m, m)
    model = OLS(X, z)
    z_predict = model(X)

    model_scikit = LinearRegression(fit_intercept = False)
    model_scikit.fit(X, z.ravel())
    z_scikit = model_scikit.predict(X)
    sk_R2 = model_scikit.score(X, z.ravel())

    sk_error = MSE(z_predict, z_scikit) # error between the scikit solution and my solution

    var_ = variance(z_predict)
    bias_ = bias(z, z_predict)

    bmin, bmax = confidence_interval(model.beta, X, var_)

    if (m == 2):
        print("Beta / confidence for m=2:")
        print(model.beta)
        print(model.beta - bmin) # print spread
        print("Sklearn coefs:")
        print(model_scikit.coef_)
        print("  m |     MSE    |    R2   | skl Err | skl R2  |")
        print("################################################")

    plot_surface(x_m, y_m, z_predict.reshape((N,N)))
    title = "m = " + str(m)
    plt.title(title)

    print("{:3d} | {:2.8f} | {:1.5f} | {:1.5f} | {:1.5f} |"
                .format(m, MSE(z, z_predict), R2(z, z_predict), sk_error, sk_R2))

    MSE_list.append(MSE(z, z_predict))
    var_list.append(var_)
    bias_list.append(bias_)

plt.figure()
plt.title("Beta confidence for highest complexity")
plt.plot(model.beta)
plt.plot(bmin)
plt.plot(bmax)

plt.figure()
plt.title("Bias-variance tradeoff")
plt.plot(m_list, MSE_list)
plt.plot(m_list, var_list)
plt.plot(m_list, bias_list)
plt.legend(["MSE", "Var", "bias"])

plt.show()
