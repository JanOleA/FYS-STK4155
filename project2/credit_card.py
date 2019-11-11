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

np.random.seed(0)

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

# shuffle X and y before splitting
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

batch_size = 100
n_batches = int(Xtrain.shape[0]/batch_size)
print(n_batches)

layers = [100, 20, 20] # does not include output layer

nn = NeuralNetwork(Xtrain, ytrain, layers, n_batches = n_batches)
nn.fit(n_epochs = 100)

"""
logReg = LogisticRegression(n_batches = n_batches)

eta_list = [1, 1e-1, 1e-2, 1e-3, 1e-4, 1e-5]
acc_list = []

for eta in eta_list:
    logReg.fit(Xtrain, ytrain, eta = eta, n_epochs = 4000)
    acc_list.append(logReg.accuracy(Xtest, ytest))
    print("Accuracy vs. test data, own logreg:", acc_list[-1])

sk_logReg = sk_LogisticRegression(solver='lbfgs')
sk_logReg.fit(Xtrain, ytrain.ravel())
y_pred = sk_logReg.predict(Xtest)
print("Accuracy vs. test data, sklearn logreg:", sk_logReg.score(Xtest, ytest))

plt.semilogx(eta_list, acc_list)
plt.show()
"""
