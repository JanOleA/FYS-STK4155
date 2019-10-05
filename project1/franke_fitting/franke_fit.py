""" Does a basic fit of the franke function using OLS regression up to fifth
order complexity.
"""

import sys
sys.path.insert(1, 'resources/')

from franke import FrankeFunction, plot_franke
from regression import *
from resources import *
import numpy as np
import matplotlib.pyplot as plt

np.random.seed(1)

N = 1000
x = np.sort(np.random.uniform(0, 1, N))
y = np.sort(np.random.uniform(0, 1, N))
x_m, y_m = np.meshgrid(x, y)

z = FrankeFunction(x_m, y_m)
z_noise = (z + np.random.normal(scale = 0.1, size = (N, N)))

plot_franke(x_m, y_m, z_noise)
plt.title("Data")

m_list = [2, 3, 4, 5]

print("  m |     MSE    |    R2   | ")
print("#############################")

for m in m_list:
    X = design_matrix(x_m, y_m, m)
    reg_fit = OLS(X, z_noise)
    z_predict = reg_fit(X)

    plot_franke(x_m, y_m, z_predict.reshape((N,N)))
    title = "m = " + str(m)
    plt.title(title)

    print("{:3d} | {:2.8f} | {:1.5f} |".format(m, MSE(z_noise, z_predict), R2(z_noise, z_predict)))

plt.show()
