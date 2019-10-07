""" Does a basic fit of the franke function using OLS regression up to fifth
order complexity, using training/test data and k-fold resampling.
"""

import sys
sys.path.insert(1, '../resources/')

from franke import FrankeFunction
from regression import *
from resources import *
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, KFold
from sklearn.preprocessing import PolynomialFeatures

np.random.seed(1)

N = 1000
x = np.sort(np.random.uniform(0, 1, N))
y = np.sort(np.random.uniform(0, 1, N))
x_m, y_m = np.meshgrid(x, y)
k = 5

z = FrankeFunction(x_m, y_m)
z_noise = (z + np.random.normal(scale = 0.3, size = (N, N)))

m_list = [2, 3, 4, 5, 7, 10, 15, 20]

kfold = KFold(n_splits = k, shuffle = True)

print("  m |     MSE    |    R2   |   var   |   bias  | ")
print("#################################################")

MSE_list = []
var_list = []
bias_list = []

for m in m_list:
    X = design_matrix(x_m, y_m, m)

    MSE_ = 0
    R2_ = 0
    var_ = 0
    bias_ = 0

    betas = np.zeros((k, int((m+1)*(m+2)/2)))

    i = 0
    for train_inds, test_inds in kfold.split(X):
        X_train = X[train_inds]
        X_test = X[test_inds]
        z_train = z_noise.ravel()[train_inds]
        z_test = z_noise.ravel()[test_inds]

        model = OLS(X_train, z_train)
        betas[i] = model.beta
        z_predict = model(X_test)

        MSE_ += MSE(z_test, z_predict)
        R2_ += R2(z_test, z_predict)
        var_ += variance(z_predict)
        bias_ += bias(z_test, z_predict)

        i += 1

    beta = np.mean(betas, axis=0)

    MSE_ /= k
    R2_ /= k
    var_ /= k
    bias_ /= k

    MSE_list.append(MSE_)
    var_list.append(var_)
    bias_list.append(bias_)

    print("{:3d} | {:2.8f} | {:1.5f} | {:1.5f} | {:1.5f} |"
                                            .format(m,
                                                    MSE_,
                                                    R2_,
                                                    var_,
                                                    bias_))

plt.figure()

plt.plot(m_list, MSE_list)
plt.plot(m_list, var_list)
plt.plot(m_list, bias_list)
plt.legend(["MSE", "Var", "bias"])
plt.show()
