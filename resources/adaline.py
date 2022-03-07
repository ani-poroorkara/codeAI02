"""
Class for 
1. Adaline with Batch Gradient Descent
2. Adaline with Stochastic Gradient Descent
3. Sigmoid Neuron
"""

import numpy as np
from numpy.random import seed


class AdaptiveLinearNeuron(object):
    def __init__(self, rate = 0.01, niter = 10):
        self.rate = rate
        self.niter = niter

    def fit(self, X, y):
        self.weight = np.zeros(1 + X.shape[1])
        self.errors = []
        self.cost = []

        for i in range(self.niter):
            output = self.net_input(X)
            errors = y - output
            self.weight[1:] += self.rate * X.T.dot(errors)
            self.weight[0] += self.rate * errors.sum()
            cost = (errors**2).sum() / 2.0
            self.cost.append(cost)
                
        return self

    def net_input(self, X):
        return np.dot(X, self.weight[1:]) + self.weight[0]

    def activation(self, X):
        return self.net_input(X)

    def predict(self, X):
        return (np.where(self.activation(X) > 0.0, 1, 0))
    
    def get_weights(self):
        print(self.weight)

    def accuracy(self, X, y):
        predicted = np.where(self.activation(X) > 0.0, 1, 0)
        correct = 0
        for i in range(len(y)):
            if y[i] == predicted[i]:
                correct += 1
        return correct / float(len(y)) * 100.0


# Stochastic Gradient Descent
class AdalineSGD(object):
    def __init__(self, rate = 0.01, niter = 10, shuffle=True, random_state=None):
        self.rate = rate
        self.niter = niter
        self.weight_initialized = False

        # If True, Shuffles training data every epoch
        self.shuffle = shuffle

        # Set random state for shuffling and initializing the weights.
        if random_state:
            seed(random_state)

    def fit(self, X, y):
        # weights
        self.initialize_weights(X.shape[1])

        # Cost function
        self.cost = []

        for i in range(self.niter):
            if self.shuffle:
                X, y = self.shuffle_set(X, y)
                cost = []
            for xi, target in zip(X, y):
                cost.append(self.update_weights(xi, target))
                avg_cost = sum(cost)/len(y)
                self.cost.append(avg_cost)
        return self

    def partial_fit(self, X, y):
        if not self.weight_initialized:
            self.initialize_weights(X.shape[1])
        if y.ravel().shape[0] > 1:
            for xi, target in zip(X, y):
                self.update_weights(xi, target)
        else:
            self.up
        return self

    def shuffle_set(self, X, y):
        r = np.random.permutation(len(y))
        return X[r], y[r]

    def initialize_weights(self, m):
        self.weight = np.zeros(1 + m)
        self.weight_initialized = True

    def update_weights(self, xi, target):
        output = self.net_input(xi)
        error = (target - output)
        self.weight[1:] += self.rate * xi.dot(error)
        self.weight[0] += self.rate * error
        cost = 0.5 * error**2
        return cost
    def get_weights(self):
        print(self.weight)
    def net_input(self, X):
        return np.dot(X, self.weight[1:]) + self.weight[0]

    def activation(self, X):
        return self.net_input(X)

    def accuracy(self, X, y):
        predicted = np.where(self.activation(X) > 0.0, 1, 0)
        correct = 0
        for i in range(len(y)):
            if y[i] == predicted[i]:
                correct += 1
        return correct / float(len(y)) * 100.0


class SigmoidNeuron(object):
    def __init__(self, rate = 0.01, niter = 10):
        self.rate = rate
        self.niter = niter
    def get_weights(self):
        print(self.weight)
        
    def sigmoid (self, x): 
        return 1/(1 + np.exp(-x)) # activation function

    def sigmoid_(self, x): 
        return x * (1 - x) # derivative of sigmoid

    def fit(self, X, y):
        self.weight = np.zeros(1 + X.shape[1])
        self.errors = []
        self.cost = []
        m = X.shape[0]

        for i in range(self.niter):
            weighted_sum = self.net_input(X)
            out = self.sigmoid(weighted_sum)
            errors = y - out
            cost = (1/(2*m)) * np.sum(errors)**2
            delta_w = self.rate * X.T.dot(np.multiply(errors,self.sigmoid_(out)))
            self.weight[0] += self.rate * errors.sum()
            self.weight[1:] += delta_w
            self.cost.append(cost)
                
        return self

    def net_input(self, X):
        return np.dot(X, self.weight[1:]) + self.weight[0]

    def activation(self, X):
        return self.net_input(X)

    def predict(self, X):
        return (np.where(self.activation(X) > 0.0, 1, 0))

    def accuracy(self, X, y):
        predicted = np.where(self.sigmoid(self.net_input(X)) > 0.0, 1, 0)
        # print(predicted)
        correct = 0
        for i in range(len(y)):
            if int(y[i]) == int(predicted[i]):
                correct += 1
        return correct / float(len(y)) * 100.0