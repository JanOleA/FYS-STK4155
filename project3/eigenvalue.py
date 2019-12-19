""" Theory provided by Yi et al, "Neural Networks Based Approach for Computing
Eigenvectors and Eigenvalues of Symmetric Matrix".

Also inspired by code and examples provided by Kristine Baluka Hein:
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

import numpy as np
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter
from resources import MSE, R2, plot_surface
import time

import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()

seed = 42

tf.set_random_seed(seed)
np.random.seed(seed)

def f(v, A):
    """ function for returning f(v) as described in the paper by Yi et al. (eq 1) """
    n = int(A.shape[0])
    I = tf.eye(n, dtype = tf.float64)

    vt = tf.transpose(v)

    return tf.matmul((tf.matmul(vt, v)*A + (1 - tf.matmul(tf.matmul(vt, A), v))*I), v)


def compute_eigenvalue(v, A):
    """ Computes the eigenvalue from the eigenvector v and matrix A """
    if len(v.shape) < 2:
        v = v[:,np.newaxis]

    vt = v.T

    num = ((vt @ A) @ v)[0,0]
    den = (vt @ v)[0,0]

    return num/den


def solve_dnn(A, smallest = False,
              t_max = 1,
              dt = 0.1,
              eps = 1e-2,
              learning_rate = 1e-4,
              num_hidden_neurons = [50],
              verbose = False):

    """ Neural network solver for the method prescribed by the Yi et al paper
    for finding the eigenpairs corresponding to the largest and smallest
    eigenvalues for a symmetric n x n matrix A.

    Inputs:
    A               : Matrix to find eigenpairs for
    smallest        : set True if trying to find the eigenvector for the
                      smallest eigenvalue (default = False)
    t_max           : Max time (default = 1)
    dt              : Time step (default = 0.1)
    eps             : Cutoff threshold. Stop iterations when error is smaller
                      than this (default = 0.01)
    learning_rate   : Learning rate of the optimizer (default = 0.0001)
    num_hidden_neurons : List containing number of neurons in hidden layers
                         (default = [50])
    verbose         : Whether or not to print info every 1000 iterations
                      (default = False)
    """
    num_hidden_layers = np.size(num_hidden_neurons)

    if (A.shape[0] != A.shape[1]):
        print(f"Matrix is not n x n, shape: {A.shape}")

    n = A.shape[0]
    N_t = int(np.ceil(t_max/dt)) + 1
    N_x = int(n)

    x_np = np.linspace(1, N_x, N_x)
    t_np = np.linspace(0, t_max, N_t)
    v0_np = np.random.rand(n) # initial guess

    if smallest:
        k = -1
    else:
        k = 1

    print(f"k = {k}")

    X, T = np.meshgrid(x_np, t_np)
    V, T = np.meshgrid(v0_np, t_np)

    x_ = (X.ravel()).reshape(-1,1)
    t_ = (T.ravel()).reshape(-1,1)
    v0_ = (V.ravel()).reshape(-1,1)

    x = tf.convert_to_tensor(x_, dtype = tf.float64)
    t = tf.convert_to_tensor(t_, dtype = tf.float64)
    v0 = tf.convert_to_tensor(v0_, dtype = tf.float64)

    points = tf.concat([x, t], 1)


    with tf.name_scope("dnn"):
        previous_layer = points

        for l in range(num_hidden_layers):
            current_layer = tf.layers.dense(previous_layer,
                                            num_hidden_neurons[l],
                                            activation = tf.nn.sigmoid)
            previous_layer = current_layer

        dnn_output = tf.layers.dense(previous_layer, 1)


    with tf.name_scope("cost"):
        trial = k*v0 + dnn_output*t

        trial_dt = tf.gradients(trial, t)

        trial = tf.reshape(trial, (N_t, N_x))
        trial_dt = tf.reshape(trial_dt, (N_t, N_x))

        cost_t = 0
        for j in range(N_t):
            # not happy about this loop, there should be a better way
            trial_t = tf.reshape(trial[j], (N_x, 1))
            trial_dt_t = tf.reshape(trial_dt[j], (N_x, 1))

            RHS = f(trial_t, A) - trial_t

            err = tf.square(trial_dt_t - RHS)
            cost_t += tf.reduce_sum(err)

        cost = tf.reduce_sum(cost_t/(N_x * N_t), name = "cost")


    with tf.name_scope("train"):
        optimizer = tf.train.AdamOptimizer(learning_rate)
        training_op = optimizer.minimize(cost)


    v_dnn = None
    init = tf.global_variables_initializer()

    with tf.Session() as sess:
        init.run()

        print(f"Initial cost: {cost.eval():g}")

        i = 0
        # run until cost threshold is reached
        while cost.eval() > eps:
            sess.run(training_op)
            i += 1

            if i%1000 == 0 and verbose:
                print(f"Training iteration {i}, cost = {cost.eval()}")

        print(f"Final cost: {cost.eval():g}")

        v_dnn = tf.reshape(trial, (N_t, N_x))
        v_dnn = v_dnn.eval()

    return v_dnn, t, i


if __name__ == "__main__":
    smallest = True # True = look for smallest eigenvalue, False = look for largest

    n = 6
    Q = np.random.rand(n,n)
    A = (Q.T + Q)/2

    ts = time.time()
    w_np, v_np = np.linalg.eig(A)
    print(f"Numpy time: {time.time() - ts}")

    A_temp = A
    if smallest:
        A_temp = -A

    print(A)

    A_tf = tf.convert_to_tensor(A_temp, dtype = tf.float64)

    max_eig = np.argmax(w_np)
    min_eig = np.argmin(w_np)

    # Neural network computation
    eps = 1e-4
    t_max = 8
    dt = 0.1
    learning_rate = 1e-3
    num_hidden_neurons = [10, 10, 10]

    ts = time.time()
    v_dnn, t, i = solve_dnn(A_tf, smallest = smallest,
                            t_max = t_max,
                            dt = dt,
                            eps = eps,
                            learning_rate = learning_rate,
                            num_hidden_neurons = num_hidden_neurons,
                            verbose = True)
    print(f"DNN time: {time.time() - ts}")

    last_v_dnn = v_dnn[-1]/np.linalg.norm(v_dnn[-1])

    print("## Numpy max ##")
    print("v =", v_np[:,max_eig])
    print("w =", w_np[max_eig])

    print("## Numpy min ##")
    print("v =", v_np[:,min_eig])
    print("w =", w_np[min_eig])

    print("## DNN ##")
    print("v =", last_v_dnn)
    print("w =", compute_eigenvalue(last_v_dnn, A))

    plt.plot(np.linspace(0, t_max, len(v_dnn)), v_dnn)

    if smallest:
        v_np_ = v_np[:,min_eig]
    else:
        v_np_ = -v_np[:,max_eig]

    for i in range(n):
        plt.plot([0, t_max], [v_np_[i], v_np_[i]], "--", color = "gray")

    plt.title("Values for v(t) as t grows")
    plt.xlabel("t")
    plt.ylabel("v_1(t), v_2(t), ... , v_n(t)")

    if smallest:
        plt.savefig("figures/eigvector_smallest.pdf")
    else:
        plt.savefig("figures/eigvector_largest.pdf")

    #plt.show()
