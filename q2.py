"""
Question 2: Part A
Data Generation
"""
from resources.data_generation import generate_random_datapoints, generate_random_datapoints_NS
from resources.perceptron import Perceptron
from resources.pocketalgorithm import PocketAlgorithm
from resources.adaline import *
from sklearn.model_selection import train_test_split
import time
import pandas as pd

X, y = generate_random_datapoints(5000, 2, 2)

"""
Question 2: Part B
Perceptron
"""
# 20% of the data
print("20% Data")
start = time.process_time()
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.8, random_state=42)
perceptron = Perceptron(2)
perceptron.train(X_train, y_train)
# perceptron.accuracy_metric()
print("Time taken: ", time.process_time() - start)

# 50% of the data
print("50% Data")
start = time.process_time()
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.5, random_state=42)
perceptron = Perceptron(2)
perceptron.train(X_train, y_train)
# perceptron.accuracy_metric()
print("Time taken: ", time.process_time() - start)

# 80% of the data
print("80% Data")
start = time.process_time()
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
perceptron = Perceptron(2)
perceptron.train(X_train, y_train)
# perceptron.accuracy_metric()
print("Time taken: ", time.process_time() - start)

# 100% of the data
print("100% Data")
start = time.process_time()
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.8, random_state=42)
perceptron = Perceptron(2)
perceptron.train(X_train, y_train)
# perceptron.accuracy_metric()
print("Time taken: ", time.process_time() - start)

"""
Question 2: Part C
Pocket Learning Algorithm
"""

#Creating data
X, y = generate_random_datapoints_NS(5000)

#initializing instances
perc = Perceptron(2)
pa = PocketAlgorithm()
aln = AdaptiveLinearNeuron()
sgd = AdalineSGD()
sig = SigmoidNeuron()

#Perceptron training
perc.train(X,y)

# Pocket learning algorithm
oneVector = np.ones((X.shape[0], 1))
X_pa = np.concatenate((oneVector, X), axis=1)
pa.train(X_pa, y)

# Data modification for adaline neurons
X1_aln= pd.DataFrame(X[:,0], columns = ['X1'])
X2_aln= pd.DataFrame(X[:,1], columns = ['X2'])
result = pd.concat([X1_aln, X2_aln], axis = 1)
df = pd.DataFrame(result, columns = ['X1', 'X2'])
dfp = np.array(df)
y_aln = np.reshape(y, (5000,))
print(dfp.shape, y_aln)

#Adaline with batch gradient descent
aln.fit(dfp, y_aln)
aln.accuracy(dfp, y_aln)

# Adaline with Stochastic gradient descent
sgd.fit(dfp, y_aln)
sgd.accuracy(dfp, y)

# Sigmoid Neuron
sig.fit(dfp, y_aln)
sig.accuracy(dfp, y_aln)