""" Regression fitting for the Franke function using OLS, Ridge and Lasso
methods. KFold cross-validation is used to calculate the MSE, bias and variance.

For all methods, various degrees of polynomials are used to do the fitting.
For Ridge and Lasso, fitting is done using various values of lambda/alpha, and
the various results are then plotted in heatmaps.
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
import seaborn as sns
sns.set()

np.random.seed(1)

N = 1000
x = np.sort(np.random.uniform(0, 1, N))
y = np.sort(np.random.uniform(0, 1, N))
x_m, y_m = np.meshgrid(x, y)
k = 5

z = FrankeFunction(x_m, y_m)
z_noise = (z + np.random.normal(scale = 0.1, size = (N, N)))

m_list = [2, 3, 4, 5, 7, 10, 15]
lambda_list = [10**(-i) for i in range(-4,4)]

kfold = KFold(n_splits = k, shuffle = True)

MSE_matrix = np.ones((len(m_list), len(lambda_list)))
R2_matrix = np.zeros((len(m_list), len(lambda_list)))
var_matrix = np.zeros((len(m_list), len(lambda_list)))
bias_matrix = np.zeros((len(m_list), len(lambda_list)))

# set up lists to hold the various results for each lambda for the best results
MSE_arrays = []
R2_arrays = []

z_select = z_noise # used to select z_noise or z
model_types = [OLS, Ridge, Lasso]
model_names = ["OLS", "Ridge", "Lasso"]

for model_type, model_name in zip(model_types, model_names):
    print("Computing with model:", model_name)
    for i, m in enumerate(m_list):
        X = design_matrix(x_m, y_m, m)
        for j, l in enumerate(lambda_list):
            MSE_ = 0
            R2_ = 0
            var_ = 0
            bias_ = 0

            print("Now computing for m = {:2d}, l = {:2.2e}".format(m, l), end="\r")
            for train_inds, test_inds in kfold.split(X):
                X_train = X[train_inds]
                X_test = X[test_inds]
                z_train = z_select.ravel()[train_inds]
                z_test = z_select.ravel()[test_inds]

                model = model_type(X_train, z_train, l)
                z_predict = model(X_test)

                MSE_ += MSE(z_test, z_predict)
                R2_ += R2(z_test, z_predict)
                var_ += variance(z_predict)
                bias_ += bias(z_test, z_predict)

            MSE_ /= k
            R2_ /= k
            var_ /= k
            bias_ /= k

            MSE_matrix[i, j] = MSE_
            R2_matrix[i, j] = R2_
            var_matrix[i, j] = var_
            bias_matrix[i, j] = bias_

            if model_name == "OLS":
                # fill the rest of the arrays that will not be filled otherwise
                # because these are the same for all lambdas with the OLS model
                MSE_matrix[i,1:] = MSE_matrix[i,0]
                R2_matrix[i,1:] = R2_matrix[i,0]
                var_matrix[i,1:] = var_matrix[i,0]
                bias_matrix[i,1:] = bias_matrix[i,0]
                break # no need to run for all lambdas for OLS

    best_MSE = np.min(MSE_matrix)
    best_R2 = np.max(R2_matrix)
    best_inds = np.unravel_index(np.argmin(MSE_matrix, axis=None), MSE_matrix.shape)
    best_m = m_list[best_inds[0]]
    best_l = lambda_list[best_inds[1]]
    MSE_arrays.append(MSE_matrix[best_inds[0]].copy()) # store the MSE for each lambda for the best complexity (measured by lowest MSE)
    R2_arrays.append(R2_matrix[best_inds[0]].copy()) # store the R2 score for each lambda for the best complexity (measured by lowest MSE)
    print("Computation complete, best MSE: {:2.4f}, best R2: {:1.4f}, best m: {:2d}, best l: {:2.2e}"
                                .format(best_MSE, best_R2, best_m, best_l))

    """ Plotting below """
    # plot MSE, bias, variance vs. complexity for some lambda
    lambda_ind = -1
    plt.figure()
    plt.plot(m_list, MSE_matrix[:,lambda_ind])
    plt.plot(m_list, bias_matrix[:,lambda_ind])
    plt.plot(m_list, var_matrix[:,lambda_ind])
    plt.legend(["MSE", "bias", "variance"])
    plt.xlabel("Complexity")
    plt.ylabel("Error")
    fn = model_name + "_bv_tradeoff.pdf"
    plt.savefig(fn)
    plt.show()

    # plot heatmaps of MSE and R2 as function of complexity and lambda
    f, ax = plt.subplots(figsize=(9, 7))
    sns.heatmap(MSE_matrix, cbar = True, square = True,
                xticklabels = lambda_list,
                yticklabels = m_list,
                annot = True, ax = ax)

    plt.title("MSE")
    plt.xlabel("lambda")
    plt.ylabel("Complexity")
    fn = model_name + "_MSE_heatmap.pdf"
    plt.savefig(fn)

    f, ax = plt.subplots(figsize=(9, 7))
    sns.heatmap(R2_matrix, cbar = True, square = True,
                xticklabels = lambda_list,
                yticklabels = m_list,
                annot = True, ax = ax)

    plt.title("R2 score")
    plt.xlabel("lambda")
    plt.ylabel("Complexity")
    fn = model_name + "_R2_heatmap.pdf"
    plt.savefig(fn)

    plt.show()

# plot MSE and R2 as a function of complexity using the best models
plt.figure()
plt.semilogx(lambda_list, MSE_arrays[0])
plt.semilogx(lambda_list, MSE_arrays[1])
plt.semilogx(lambda_list, MSE_arrays[2])
plt.legend(model_names)
plt.title("MSE value")
plt.xlabel("lambda")
plt.ylabel("MSE")
plt.xlim((lambda_list[0], lambda_list[-1]))
plt.savefig("mse_scores.pdf")
plt.figure()
plt.semilogx(lambda_list, R2_arrays[0])
plt.semilogx(lambda_list, R2_arrays[1])
plt.semilogx(lambda_list, R2_arrays[2])
plt.legend(model_names)
plt.title("R2 scores")
plt.xlabel("lambda")
plt.ylabel("R2")
plt.xlim((lambda_list[0], lambda_list[-1]))
plt.savefig("R2_scores.pdf")
plt.show()
