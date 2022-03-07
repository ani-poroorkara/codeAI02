"""
"""

import numpy as np

class PocketAlgorithm(object):

    def __init__(self, iteration = 1000 ,misClassifications=1, minMisclassifications=10000, learningRate=0.01):
        self.learningRate = learningRate
        self.weights = np.random.rand(3, 1)
        self.iteration = iteration
        self.misClassifications = misClassifications
        self.minMisclassifications = minMisclassifications

    def accuracy_metric(self, X_train):
        print("Minimum Misclassifications : ", self.minMisclassifications)
        print("Best weight is: ", self.weights.transpose())
        print("Best Case Accuracy of Pocket Learning Algorithm is: ",(((X_train.shape[0]-self.minMisclassifications)/X_train.shape[0])*100),"%")
    
    def train(self, X_train, y):
        iteration = 0
        while (self.misClassifications != 0 and (iteration<self.iteration)):
            iteration += 1
            self.misClassifications = 0

            for i in range(0, len(X_train)):
                currentX = X_train[i].reshape(-1, X_train.shape[1])
                currentY = y[i]
                wTx = np.dot(currentX, self.weights)[0][0]

                if currentY == 1 and wTx < 0:
                    self.misClassifications += 1
                    self.weights = self.weights + self.learningRate * np.transpose(currentX)
                elif currentY == 0 and wTx > 0:
                    self.misClassifications += 1
                    self.weights = self.weights - self.learningRate * np.transpose(currentX)
            # plotData.append(misClassifications)
            
            if self.misClassifications<self.minMisclassifications:
                self.minMisclassifications = self.misClassifications

            # self.accuracy_metric(X_train)