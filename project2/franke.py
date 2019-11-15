import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression as sk_LinearRegression

sys.path.insert(1, 'resources/')
from resources import *
from neural_network import NeuralNetworkLinear # own neural network

# color list for plotting
color_list = ["green", "blue", "red", "cyan", "purple",
              "orange", "sage", "brown", "black"]

plt.rcParams.update({'font.size': 16})

# Setting up features and targets
x1 = np.linspace(0, 1, 100)
x2 = np.linspace(0, 1, 100)

m_x1, m_x2 = np.meshgrid(x1, x2)

y_original = FrankeFunction(m_x1, m_x2)
noise = np.random.normal(scale = 0.1, size = y_original.shape)
y = y_original + noise

f1 = np.ravel(m_x1)
f2 = np.ravel(m_x2)

X = np.column_stack((f1, f2))

y = np.ravel(y)
y = y[:,np.newaxis]

trainingShare = 0.8
seed = 1
Xtrain, Xtest, ytrain, ytest = train_test_split(X, y, train_size = trainingShare,
                                                test_size = 1 - trainingShare,
                                                random_state = seed)

layers = [50, 20] # hidden layers

params = False
if len(sys.argv) > 1:
    if sys.argv[1] == "params":
        params = True

""" Testing various learning rates """
if params:
    """ Run parameter test """
    etas = [10**(-i) for i in range(5)]

    accuracys_train = []
    costs_train = []
    accuracys_test = []
    costs_test = []

    batch_size = 50
    lmbda = 0

    for eta in etas:
        nn = NeuralNetworkLinear(Xtrain, ytrain, layers,
                                 Xtest = Xtest, ytest = ytest)
        n_batches = int(Xtrain.shape[0]/batch_size)
        a, b, c, d = nn.fit(n_epochs = 100, eta = eta,
                            n_batches = n_batches, lmbda = lmbda)

        accuracys_train.append(a)
        costs_train.append(b)
        accuracys_test.append(c)
        costs_test.append(d)

        print("Test accuracy:", nn.accuracy(Xtest, ytest)) # test accuracy after training

    plt.figure(figsize=(10,8))
    plt.title("Accuracy score for varying learning rate")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    for i, eta in enumerate(etas):
        color = color_list[i]
        plt.plot(accuracys_train[i], "--", color = color,
                 label="Train, eta = {:g}".format(eta))
        plt.plot(accuracys_test[i], color = color,
                 label="Test, eta = {:g}".format(eta))

    plt.legend(loc="lower right")
    plt.savefig("results/accuracy_eta_franke.pdf")

    plt.figure(figsize=(10,8))
    plt.title("Cost function for varying learning rate")
    plt.xlabel("Epoch")
    plt.ylabel("Cost")
    for i, eta in enumerate(etas):
        color = color_list[i]
        plt.plot(costs_train[i], "--", color = color,
                 label="Train, eta = {:g}".format(eta))
        plt.plot(costs_test[i], color = color,
                 label="Test, eta = {:g}".format(eta))

    plt.legend(loc="upper right")
    plt.savefig("results/cost_eta_franke.pdf")

    """ Testing various batch sizes with learning rate 0.1 """

    accuracys_train = []
    costs_train = []
    accuracys_test = []
    costs_test = []

    eta = 0.1
    batch_sizes = [10, 50, 100, 500, 1000]

    for batch_size in batch_sizes:
        nn = NeuralNetworkLinear(Xtrain, ytrain, layers,
                                 Xtest = Xtest, ytest = ytest)
        n_batches = int(Xtrain.shape[0]/batch_size)
        a, b, c, d = nn.fit(n_epochs = 100, eta = eta,
                            n_batches = n_batches, lmbda = lmbda)

        accuracys_train.append(a)
        costs_train.append(b)
        accuracys_test.append(c)
        costs_test.append(d)

        print("Test accuracy:", nn.accuracy(Xtest, ytest)) # test accuracy after training

    plt.figure(figsize=(10,8))
    plt.title("Accuracy score for varying batch size")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    for i, batch_size in enumerate(batch_sizes):
        color = color_list[i]
        plt.plot(accuracys_train[i], "--", color = color,
                 label="Train, batch size = {:d}".format(batch_size))
        plt.plot(accuracys_test[i], color = color,
                 label="Test, batch size = {:d}".format(batch_size))

    plt.legend(loc="lower right")
    plt.savefig("results/accuracy_batch_franke.pdf")

    plt.figure(figsize=(10,8))
    plt.title("Cost function for varying batch size")
    plt.xlabel("Epoch")
    plt.ylabel("Cost")
    for i, batch_size in enumerate(batch_sizes):
        color = color_list[i]
        plt.plot(costs_train[i], "--", color = color,
                 label="Train, batch size = {:d}".format(batch_size))
        plt.plot(costs_test[i], color = color,
                 label="Test, batch size = {:d}".format(batch_size))

    plt.legend(loc="upper right")
    plt.savefig("results/cost_batch_franke.pdf")

    """ Testing various regularization params with learning rate 0.1,
    batch size 100 """

    accuracys_train = []
    costs_train = []
    accuracys_test = []
    costs_test = []

    eta = 0.1
    batch_size = 100
    regularization_params = [10**(-i) for i in range(5)]
    regularization_params.append(0)

    for lmbda in regularization_params:
        nn = NeuralNetworkLinear(Xtrain, ytrain, layers,
                                 Xtest = Xtest, ytest = ytest)
        n_batches = int(Xtrain.shape[0]/batch_size)
        a, b, c, d = nn.fit(n_epochs = 100, eta = eta,
                            n_batches = n_batches, lmbda = lmbda)

        accuracys_train.append(a)
        costs_train.append(b)
        accuracys_test.append(c)
        costs_test.append(d)

        print("Test accuracy:", nn.accuracy(Xtest, ytest)) # test accuracy after training

    plt.figure(figsize=(10,8))
    plt.title("Accuracy score for varying reg. parameters, batch size = 100")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    for i, lmbda in enumerate(regularization_params):
        color = color_list[i]
        plt.plot(accuracys_train[i], "--", color = color,
                 label="Train, lambda = {:g}".format(lmbda))
        plt.plot(accuracys_test[i], color = color,
                 label="Test, lambda = {:g}".format(lmbda))

    plt.legend(loc="lower right")
    plt.savefig("results/accuracy_lmbda_franke.pdf")

    plt.figure(figsize=(10,8))
    plt.title("Cost function for varying reg. parameters, batch size = 100")
    plt.xlabel("Epoch")
    plt.ylabel("Cost")
    for i, lmbda in enumerate(regularization_params):
        color = color_list[i]
        plt.plot(costs_train[i], "--", color = color,
                 label="Train, lambda = {:g}".format(lmbda))
        plt.plot(costs_test[i], color = color,
                 label="Test, lambda = {:g}".format(lmbda))

    plt.legend(loc="upper right")
    plt.savefig("results/cost_lmbda_franke.pdf")
    plt.show()
else:
    """ if not parameter test, run with some good parameters and plot fit """
    batch_size = 100
    eta = 0.01
    lmbda = 1e-4
    nn = NeuralNetworkLinear(Xtrain, ytrain, layers)
    n_batches = int(Xtrain.shape[0]/batch_size)
    nn.fit(n_epochs = 100, eta = eta, n_batches = n_batches, lmbda = lmbda)

    plot_surface(m_x1, m_x2, y_original)
    plt.title("Original data without noise")
    plot_surface(m_x1, m_x2, y.reshape(100,100))
    plt.title("Original data with noise")

    y_predict = nn.predict(X).reshape(100,100)

    plot_surface(m_x1, m_x2, y_predict, show = True)
    plt.title("Neural network prediction")

    print("Test accuracy:", nn.accuracy(Xtest, ytest)) # test accuracy after training
