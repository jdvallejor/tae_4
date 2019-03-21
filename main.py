from sklearn.externals import joblib
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.neural_network import MLPClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
import numpy as np
import time
import sys

# Reading data
print("Stage 1: Reading data and normalizing values\n")
train_data = np.genfromtxt('data/mnist_train.csv', delimiter=',')

# Train data
X_train = train_data[1:, 1:]
Y_train = train_data[1:, 0:1]
Y_train = Y_train.flatten()

# test data
test_data = np.genfromtxt('data/mnist_test.csv', delimiter=',')

X_test = test_data[:,1:]
Y_test = test_data[:,0:1]
Y_test = Y_test.flatten()

# Normalizing data
X_train = X_train / 255
X_test = X_test / 255

# Cast labels to string
new_array = np.array(
    ["%.0f" % x for x in Y_train.reshape(Y_train.size)])
Y_train = new_array.reshape(Y_train.shape)

new_array = np.array(
    ["%.0f" % x for x in Y_test.reshape(Y_test.size)])
Y_test = new_array.reshape(Y_test.shape)
print("--------------------------------------------------------------------------------")

# MULTINOMIAL LOGISTIC REGRESSION
print("Stage 2: Training multinomial logistic regression:")
t0 = time.time()
train_samples = 60000

# Defining model
clf = LogisticRegression(C=50. / train_samples,
                         multi_class='multinomial',
                         penalty='l1', solver='saga', tol=0.1)

# Training model
clf.fit(X_train, Y_train)
# Testing model
score = clf.score(X_test, Y_test)

# Results
print("Training set score: %.4f" % score)
run_time = time.time() - t0
print('Stage 2 run in %.3f s' % run_time)
joblib.dump(clf, "models/Logistic_model.pkl", compress=3)
print("--------------------------------------------------------------------------------")

# MULTINOMIAL SUPPORT VECTOR MACHINE
print("Stage 3: Training multinomial support vector machine:")
t0 = time.time()

# Definig model
clf = LinearSVC()

# Training model
clf.fit(X_train, Y_train)
# Testing model
score = clf.score(X_test, Y_test)

print("Test set score: %4f" % score)
run_time = time.time() - t0
print('Stage 3 run in %.3f s' % run_time)
joblib.dump(clf, "models/SVM_model.pkl", compress=3)
print("--------------------------------------------------------------------------------")

# NEURAL NETWORK
print("Stage 4: Training neural network:")
t0 = time.time()

# Definig model
mlp = MLPClassifier(hidden_layer_sizes=(50,), max_iter=40, alpha=1e-4,
                    solver='sgd', verbose=10, tol=1e-4, random_state=1,
                    learning_rate_init=.1)

# Training model
mlp.fit(X_train, Y_train)
# Testing model
score = mlp.score(X_test, Y_test)

print("Test set score: %4f" % score)
run_time = time.time() - t0
print('Stage 4 run in %.3f s' % run_time)
joblib.dump(mlp, "models/NeuralNetwork.pkl", compress=3)
print("--------------------------------------------------------------------------------")

# DECISION TREE
print("Stage 5: Training decision tree:")
t0 = time.time()

# Definig model
clf = DecisionTreeClassifier()

# Training model
clf.fit(X_train, Y_train)
# Testing model
score = clf.score(X_test, Y_test)

print("Test set score: %4f" % score)
run_time = time.time() - t0
print('Stage 5 run in %.3f s' % run_time)
joblib.dump(clf, "models/DecisionTree.pkl", compress=3)
print("--------------------------------------------------------------------------------")

# RANDOM FOREST
print("Stage 6: Training random forest:")
t0 = time.time()

# Definig model
clf = RandomForestClassifier(n_estimators=10)

# Training model
clf.fit(X_train, Y_train)
# Testing model
score = clf.score(X_test, Y_test)

print("Test set score: %4f" % score)
run_time = time.time() - t0
print('Stage 6 run in %.3f s' % run_time)
joblib.dump(clf, "models/RandomForest.pkl", compress=3)
print("--------------------------------------------------------------------------------")

