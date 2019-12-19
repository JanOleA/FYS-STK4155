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
from nn_solver import solve_dnn
import time
import seaborn as sns
sns.set()

import tensorflow.compat.v1 as tf

tf.disable_v2_behavior()
tf.reset_default_graph()
tf.set_random_seed(4155)

dx = 0.02
dt = 0.02
num_hidden_neurons = [30, 10]

learning_rates = [1, 0.5, 0.1, 0.05, 0.01, 0.005, 0.001]
num_iters = [500, 1000, 2000, 4000, 6000, 8000, 10000]

MSE_array = np.empty((len(learning_rates), len(num_iters)))
time_array = np.empty(np.shape(MSE_array))

for i, learning_rate in enumerate(learning_rates):
    for j, num_iter in enumerate(num_iters):
        print("### Solving using DNN ###")
        ts = time.time()
        u_dnn, u_analytic, x, t = solve_dnn(dx = dx, dt = dt,
                                            learning_rate = learning_rate,
                                            num_iter = num_iter,
                                            num_hidden_neurons = num_hidden_neurons)
        time_elapsed = time.time() - ts
        print(f"LR: {learning_rate}, iter: {num_iter}, DNN time used: {time_elapsed}s")
        print(f"Total MSE for DNN: {MSE(u_analytic, u_dnn)}")
        MSE_array[i,j] = MSE(u_analytic, u_dnn)
        time_array[i,j] = time_elapsed

np.save("data/MSE_array.npy", MSE_array)
np.save("data/time_array.npy", time_array)

plt.figure()
sns.heatmap(MSE_array, xticklabels = num_iters, yticklabels = learning_rates,
            cbar = True)
plt.savefig("figures/mse_array.pdf")
plt.figure()
sns.heatmap(time_array, xticklabels = num_iters, yticklabels = learning_rates,
            cbar = True)
plt.savefig("figures/time_array.pdf")
#plt.show()
