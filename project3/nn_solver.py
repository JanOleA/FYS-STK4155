""" Inspired by code and examples provided by Kristine Baluka Hein:
"Data Analysis and Machine Learning: Using Neural networks to solve ODEs and PDEs":
https://compphysics.github.io/MachineLearning/doc/pub/odenn/html/._odenn-bs000.html

"Example: Solving the diffusion equation":
https://github.com/krisbhei/DEnet/blob/master/DNN_Diffeq/example_pde_diffusion.ipynb
"""

import warnings
warnings.filterwarnings('ignore',category=FutureWarning)
""" disabled some warnings because of warning spam with numpy 1.17+ and TF
(https://github.com/tensorflow/tensorflow/issues/30427)
"""

from analytical import analytical_solution
import numpy as np
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter
from resources import MSE, R2, plot_surface
from finite_diff import solve
import time

import tensorflow.compat.v1 as tf

tf.disable_v2_behavior()
tf.reset_default_graph()
tf.set_random_seed(4155)


def solve_dnn(dx, dt = None, learning_rate = 0.001, num_iter = 1000, num_hidden_neurons = [50]):
    """ Solver for a specific case of the diffusion equation using a neural
    network to compute the solution of the equation:
    u_t = u_(xx), t > 0, x in [0, 1]
    u(x, 0) = sin(pi*x), 0 < x < 1
    u(0, t) = 0, u(1, t) = 0, t >= 0

    Input:
    dx                 : spatial difference
    dt                 : time difference, if None, uses the stability criterion
                         of the finite difference method (i.e. dt/dx^2 <= 0.5)
    learning_rate      : learning rate of the optimizer (default = 0.001)
    num_iter           : number of NN iterations (default = 1000)
    num_hidden_neurons : list containing number of
                         neurons for each hidden layer (default = [50])
    """
    tf.reset_default_graph()
    num_hidden_layers = np.size(num_hidden_neurons)

    if dt == None:
        dt = 0.5*dx**2
    N_t = int(np.ceil(1/dt)) + 1
    N_x = 1/dx + 1

    if N_x%1 != 0:
        print("Must be possible to make integer number of spatial points")
        print(f"Was given dx = {dx} which gives N_x = {N_x}")
        sys.exit(1)

    N_x = int(N_x)

    print(f"Neural network N_x = {N_x}, N_t = {N_t}")

    x_np = np.linspace(0, 1, N_x)
    t_np = np.linspace(0, 1, N_t)

    X, T = np.meshgrid(x_np, t_np)

    x_ = (X.ravel()).reshape(-1,1)
    t_ = (T.ravel()).reshape(-1,1)

    x = tf.convert_to_tensor(x_)
    t = tf.convert_to_tensor(t_)

    points = tf.concat([x, t], 1)

    # main NN iteration
    with tf.variable_scope("dnn"):
        previous_layer = points

        for l in range(num_hidden_layers):
            current_layer = tf.layers.dense(previous_layer,
                                            num_hidden_neurons[l],
                                            activation = tf.nn.sigmoid)
            previous_layer = current_layer

        dnn_output = tf.layers.dense(previous_layer, 1)


    initial = tf.sin(np.pi*x)
    # cost function
    with tf.name_scope("cost"):
        u_trial = (1 - t)*initial + x*(1 - x)*t*dnn_output

        u_trial_dt = tf.gradients(u_trial, t)
        u_trial_d2x = tf.gradients(tf.gradients(u_trial, x), x)

        err = tf.square(u_trial_dt[0] - u_trial_d2x[0])
        cost = tf.reduce_sum(err, name = 'cost')


    # training
    with tf.name_scope('train'):
        optimizer = tf.train.AdamOptimizer(learning_rate)
        training_op = optimizer.minimize(cost)


    u_analytic = tf.sin(np.pi*x)*tf.exp(-np.pi**2*t)
    u_dnn = None

    init = tf.global_variables_initializer()


    with tf.Session() as sess:
        init.run()

        for i in range(num_iter):
            sess.run(training_op)

        u_analytic = u_analytic.eval()
        u_dnn = u_trial.eval()

    u_dnn = u_dnn.reshape((N_t, N_x))
    u_analytic = u_analytic.reshape((N_t, N_x))

    return u_dnn, u_analytic, x_np, t_np


def MSE_FD(u_finite_diff, fd_t, x):
    """ Function which calculates the analytical solution and MSE for the
    finite difference solution at given time points fd_t and positions x

    Separate function is used in order to recalculate u_analytic for this
    specific case
    """
    X, T = np.meshgrid(x, fd_t)
    u_analytic = analytical_solution(X, T)

    return MSE(u_analytic, u_finite_diff)


def R2_FD(u_finite_diff, fd_t, x):
    """ Function which calculates the analytical solution and R2 score for the
    finite difference solution at given time points fd_t and positions x

    Separate function is used in order to recalculate u_analytic for this
    specific case
    """
    X, T = np.meshgrid(x, fd_t)
    u_analytic = analytical_solution(X, T)

    return R2(u_analytic, u_finite_diff)


if __name__ == "__main__":
    dx = 0.02
    dt = 0.02
    learning_rate = 0.05
    num_iter = 3000
    num_hidden_neurons = [30, 10]

    print("### Solving using DNN ###")
    ts = time.time()
    u_dnn, u_analytic, x, t = solve_dnn(dx = dx, dt = dt,
                                        learning_rate = learning_rate,
                                        num_iter = num_iter,
                                        num_hidden_neurons = num_hidden_neurons)
    time_elapsed = time.time() - ts
    print(f"DNN time used: {time_elapsed}s")

    print("### Solving using finite difference ###")
    ts = time.time()
    u_finite_diff, fd_t = solve(dx, 1)
    time_elapsed = time.time() - ts
    print(f"FD time used: {time_elapsed}s")

    skip_points = int(np.round(dt/(fd_t[1] - fd_t[0])))
    """ Finite diff calculates more points, so remove in-between points
    to only compare for the points that are calculated in the NN version.

    Because of numerical errors, these might not be _exactly_ at the same
    time values as the calculations from the neural network, so the MSE and R2
    must be calculated using a separate analytical solution with the correct
    time values
    """
    u_finite_diff = u_finite_diff[::skip_points]
    fd_t = fd_t[::skip_points]

    X, T = np.meshgrid(x, t)

    diff = np.abs(u_analytic - u_dnn)
    print("Max absolute difference between analytical solution and DNN =", np.max(diff))
    print(f"Total MSE for DNN:               {MSE(u_analytic, u_dnn)}")
    print(f"Total MSE for finite difference: {MSE_FD(u_finite_diff, fd_t, x)}")

    print(f"Total R2 for DNN:               {R2(u_analytic, u_dnn)}")
    print(f"Total R2 for finite difference: {R2_FD(u_finite_diff, fd_t, x)}")

    fig = plt.figure(figsize=(10, 10))
    ax = fig.gca(projection="3d")
    ax.set_title(f"DNN solution, dx = {dx}, hidden neurons: [" + ", ".join(map(str, num_hidden_neurons)) + "]"
                 + f", learning rate = {learning_rate}, iterations = {num_iter}")

    surf = ax.plot_surface(X, T, u_dnn, linewidth = 0, antialiased = False,
                           cmap = cm.viridis)
    ax.set_xlabel("Position $x$")
    ax.set_ylabel("Time $t$")
    ax.view_init(elev=5., azim=5)
    fig.savefig("figures/dnn.pdf")

    fig = plt.figure(figsize=(10, 10))
    ax = fig.gca(projection="3d")
    ax.set_title(f"Analytic solution, dx = {dx}, hidden neurons: [" + ", ".join(map(str, num_hidden_neurons)) + "]"
                 + f", learning rate = {learning_rate}, iterations = {num_iter}")

    surf = ax.plot_surface(X, T, u_analytic, linewidth = 0, antialiased = False,
                           cmap = cm.viridis)
    ax.set_xlabel("Position $x$")
    ax.set_ylabel("Time $t$")
    ax.view_init(elev=5., azim=5)
    fig.savefig("figures/analytic.pdf")

    fig = plt.figure(figsize=(10, 10))
    ax = fig.gca(projection="3d")
    ax.set_title(f"Difference, dx = {dx}, hidden neurons: [" + ", ".join(map(str, num_hidden_neurons)) + "]"
                 + f", learning rate = {learning_rate}, iterations = {num_iter}")

    surf = ax.plot_surface(X, T, diff, linewidth = 0, antialiased = False,
                           cmap = cm.viridis)
    ax.set_xlabel("Position $x$")
    ax.set_ylabel("Time $t$")
    ax.view_init(elev=5., azim=5)
    fig.savefig("figures/diff.pdf")

    plt.figure(figsize=(14, 10))
    N_t = len(t)
    for i, j in enumerate(np.linspace(0, N_t-1, 8)):
        j = int(j)
        t_ = t[j]
        dnn_solution = u_dnn[j]
        ana_solution = u_analytic[j]
        fdf_solution = u_finite_diff[j]

        plt.subplot(241 + i)
        plt.plot(x, fdf_solution, ls = "dashed", label = "finite difference solution")
        plt.plot(x, dnn_solution, label = "dnn solution")
        plt.plot(x, ana_solution, label = "analytical solution")
        plt.ylim((0,1.2))
        plt.title(f"t = {t_:.3f}, dx = {dx}")
        plt.xlabel("x")
        plt.ylabel("u(x,t)")
        plt.legend()

    plt.subplots_adjust(left = 0.05, right = 0.95, wspace = 0.3, hspace = 0.3)
    plt.savefig("figures/multiple_dnn.pdf")

    plt.show()
