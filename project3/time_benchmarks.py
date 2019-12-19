""" Inspired by code and examples provided by Kristine Baluka Hein:
"Data Analysis and Machine Learning: Using Neural networks to solve ODEs and PDEs":
https://compphysics.github.io/MachineLearning/doc/pub/odenn/html/._odenn-bs000.html

"Example: Solving the diffusion equation":
https://github.com/krisbhei/DEnet/blob/master/DNN_Diffeq/example_pde_diffusion.ipynb
"""

import warnings
warnings.filterwarnings('ignore',category=FutureWarning)
warnings.filterwarnings('ignore',category=Warning)
""" disabled some warnings because of warning spam with numpy 1.17+ and TF
(https://github.com/tensorflow/tensorflow/issues/30427)
"""

from analytical import analytical_solution
import numpy as np
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter
from resources import MSE, R2
from nn_solver import solve_dnn, MSE_FD
import time
from finite_diff import solve

import tensorflow.compat.v1 as tf

font = {'size'   : 14}
plt.rc('font', **font)

tf.disable_v2_behavior()
tf.reset_default_graph()
tf.set_random_seed(4155)

dx_values = [0.1, 0.05, 0.01] #, 0.005, 0.001]

learning_rate = 0.01
num_iter = 2000

num_hidden_neurons = [30, 10]

NN = np.empty((2, len(dx_values)))
FD = np.empty((2, len(dx_values)))

for i, dx in enumerate(dx_values):
    dt_nn = dx

    print("### Solving using DNN ###")
    ts = time.time()
    u_dnn, u_analytic, x, t = solve_dnn(dx = dx, dt = dt_nn,
                                        learning_rate = learning_rate,
                                        num_iter = num_iter,
                                        num_hidden_neurons = num_hidden_neurons)
    time_elapsed = time.time() - ts
    print(f"dx = {dx}, DNN time used: {time_elapsed}s")
    error_nn = MSE(u_analytic, u_dnn)
    NN[0, i] = time_elapsed
    NN[1, i] = error_nn

    print("### Solving using finite difference ###")
    ts = time.time()
    u_finite_diff, fd_t = solve(dx, 1)
    time_elapsed_fd = time.time() - ts
    print(f"FD time used: {time_elapsed_fd}s")
    error_fd = MSE_FD(u_finite_diff, fd_t, x)
    FD[0, i] = time_elapsed_fd
    FD[1, i] = error_fd

plt.figure(figsize=(10,10))
plt.subplot(211)
plt.semilogy(dx_values, FD[0])
plt.title("Finite difference time evolution")
plt.xlabel("dx")
plt.ylabel("time [s]")
plt.subplot(212)
plt.plot(dx_values, FD[1])
plt.title("Finite difference error evolution")
plt.xlim((dx[-1], dx[0]))
plt.xlabel("dx")
plt.ylabel("MSE")

plt.figure(figsize=(10,10))
plt.subplot(211)
plt.semilogy(dx_values, NN[0])
plt.title("Neural network time evolution")
plt.xlabel("dx")
plt.ylabel("time [s]")
plt.subplot(212)
plt.plot(dx_values, NN[1])
plt.title("Neural network error evolution")
plt.xlim((dx[-1], dx[0]))
plt.xlabel("dx")
plt.ylabel("MSE")

plt.show()
