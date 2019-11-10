import sys
import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.metrics import confusion_matrix, accuracy_score, roc_auc_score
from sklearn.linear_model import LogisticRegression as sk_LogisticRegression

sys.path.insert(1, 'resources/')
from resources import *
from logreg import LogisticRegression # own logistic regression

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

logReg = LogisticRegression(n_batches = 10)

logReg.fit(Xtrain, ytrain, eta = 1e-0, n_epochs = 2000)
print("Accuracy vs. test data, own logreg:", logReg.accuracy(Xtest, ytest))

sk_logReg = sk_LogisticRegression(solver='lbfgs')
sk_logReg.fit(Xtrain, ytrain.ravel())
y_pred = sk_logReg.predict(Xtest)
print("Accuracy vs. test data, sklearn logreg:", sk_logReg.score(Xtest, ytest))
