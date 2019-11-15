import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.metrics import confusion_matrix, accuracy_score, roc_auc_score
from sklearn.linear_model import LogisticRegression as sk_LogisticRegression

sys.path.insert(1, 'resources/')
from resources import *
from logreg import LogisticRegression # own logistic regression
from neural_network import NeuralNetwork # own neural network

if len(sys.argv) < 2:
    print("Please provide keyword argument ('nn' for neural network, 'log' for logistic regression)")
    print("Defaulting to logistic regression")
    mode = "logreg"
else:
    if sys.argv[1].lower() == "nn":
        mode = "nn"
    elif sys.argv[1].lower() == "log":
        mode = "logreg"
    else:
        print("Incorrect keyword argument, use 'nn' for neural network or 'log' for logistic regression")
        sys.exit(1)

np.random.seed(1)

plt.rcParams.update({'font.size': 16})

filename = 'data/default of credit card clients.xls'

nanDict = {}
df = pd.read_excel(filename, header=1, skiprows=0, index_col=0, na_values=nanDict)
df.rename(index=str, columns={"default payment next month": "defaultPaymentNextMonth"}, inplace=True)

# Remove instances with zeros only for past bill statements or paid amounts
df = df.drop(df[(df.BILL_AMT1 == 0) &
                (df.BILL_AMT2 == 0) &
                (df.BILL_AMT3 == 0) &
                (df.BILL_AMT4 == 0) &
                (df.BILL_AMT5 == 0) &
                (df.BILL_AMT6 == 0)].index)

df = df.drop(df[(df.PAY_AMT1 == 0) &
                (df.PAY_AMT2 == 0) &
                (df.PAY_AMT3 == 0) &
                (df.PAY_AMT4 == 0) &
                (df.PAY_AMT5 == 0) &
                (df.PAY_AMT6 == 0)].index)

# remove instances with unknown pay value
df = df.drop(df[(df.PAY_0 == -2) |
                (df.PAY_2 == -2) |
                (df.PAY_3 == -2) |
                (df.PAY_4 == -2) |
                (df.PAY_5 == -2) |
                (df.PAY_6 == -2)].index)

# Features and targets
X = df.loc[:, df.columns != 'defaultPaymentNextMonth'].values
y = df.loc[:, df.columns == 'defaultPaymentNextMonth'].values

print(df.head())

# Categorical variables to one-hot's
onehotencoder = OneHotEncoder(categories="auto", sparse="false")
scaler = StandardScaler(with_mean=False)

X = ColumnTransformer([("", onehotencoder, [1,2,3,5,6,7,8,9]),],
                      remainder="passthrough").fit_transform(X)

X = scaler.fit_transform(X)
if mode == "nn": y = onehotencoder.fit_transform(y).toarray()

# shuffle X and y
rand_ind = np.arange(X.shape[0])
np.random.shuffle(rand_ind)
X = X[rand_ind,:]
y = y[rand_ind,:]

# Train-test split
trainingShare = 0.8
seed = 1
Xtrain, Xtest, ytrain, ytest = train_test_split(X, y, train_size = trainingShare,
                                                test_size = 1 - trainingShare,
                                                random_state = seed)

# Input Scaling
sc = StandardScaler()
Xtrain = sc.fit_transform(Xtrain)
Xtest = sc.transform(Xtest)

# color list for plotting
color_list = ["green", "blue", "red", "cyan", "purple",
              "orange", "sage", "brown", "black"]

if mode == "nn":
    layers = [50, 20] # hidden layers

    """ Testing various learning rates """

    etas = [10**(-i) for i in range(5)]

    accuracys_train = []
    costs_train = []
    accuracys_test = []
    costs_test = []

    batch_size = 50
    lmbda = 0

    for eta in etas:
        nn = NeuralNetwork(Xtrain, ytrain, layers,
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
    plt.savefig("results/accuracy_eta.pdf")

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
    plt.savefig("results/cost_eta.pdf")

    """ Testing various batch sizes with learning rate 0.1 """

    accuracys_train = []
    costs_train = []
    accuracys_test = []
    costs_test = []

    eta = 0.1
    batch_sizes = [10, 50, 100, 500, 1000]

    for batch_size in batch_sizes:
        nn = NeuralNetwork(Xtrain, ytrain, layers,
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
    plt.savefig("results/accuracy_batch.pdf")

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
    plt.savefig("results/cost_batch.pdf")

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
        nn = NeuralNetwork(Xtrain, ytrain, layers,
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
    plt.savefig("results/accuracy_lmbda.pdf")

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
    plt.savefig("results/cost_lmbda.pdf")
    plt.show()

if mode == "logreg":
    batch_size = 100
    n_batches = int(Xtrain.shape[0]/batch_size)
    logReg = LogisticRegression(n_batches = n_batches, allow_early_stop = False)

    etas = [1, 1e-1, 1e-2, 1e-3, 1e-4, 1e-5]
    acc_list = []

    accuracys_train = []
    costs_train = []
    accuracys_test = []
    costs_test = []

    for eta in etas:
        a, b, c, d = logReg.fit(Xtrain, ytrain, eta = eta, n_epochs = 2000,
                                Xtest = Xtest, ytest = ytest)
        acc_list.append(logReg.accuracy(Xtest, ytest))

        accuracys_train.append(a)
        costs_train.append(b)
        accuracys_test.append(c)
        costs_test.append(d)

        print("Accuracy vs. test data, own logreg:", acc_list[-1])

    plt.figure(figsize=(10,8))
    plt.title("Accuracy score for varying learning rate, logistic regression")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    for i, eta in enumerate(etas):
        color = color_list[i]
        plt.plot(accuracys_train[i], "--", color = color,
                 label="Train, eta = {:g}".format(eta))
        plt.plot(accuracys_test[i], color = color,
                 label="Test, eta = {:g}".format(eta))

    plt.legend(loc="lower right")
    plt.savefig("results/accuracy_eta_logreg.pdf")

    plt.figure(figsize=(10,8))
    plt.title("Cost function for varying learning rate, logistic regression")
    plt.xlabel("Epoch")
    plt.ylabel("Cost")
    for i, eta in enumerate(etas):
        color = color_list[i]
        plt.plot(costs_train[i], "--", color = color,
                 label="Train, eta = {:g}".format(eta))
        plt.plot(costs_test[i], color = color,
                 label="Test, eta = {:g}".format(eta))

    plt.legend(loc="upper right")
    plt.savefig("results/cost_eta_logreg.pdf")

    sk_logReg = sk_LogisticRegression(solver='lbfgs')
    sk_logReg.fit(Xtrain, ytrain.ravel())
    y_pred = sk_logReg.predict(Xtest)
    print("Accuracy vs. test data, sklearn logreg:", sk_logReg.score(Xtest, ytest))

    plt.figure()
    plt.semilogx(etas, acc_list)
    plt.show()
