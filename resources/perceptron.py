"""
Question 2: Part B
Perceptron Learning Algorithm. 
"""

import numpy as np

class Perceptron(object):

    def __repr__(self) -> str:
        return 'Object<Perceptron>: ' + self.weights

    def __init__(self, no_of_inputs, threshold=5, learning_rate=0.01):
        self.threshold = threshold
        self.learning_rate = learning_rate
        self.weights = np.zeros(no_of_inputs + 1)
           
    def predict(self, inputs):
        summation = np.dot(inputs, self.weights[1:]) + self.weights[0]
        if summation > 0:
          activation = 1
        else:
          activation = 0            
        return activation

    def train(self, training_inputs, labels):
        # print("Predicted \tExepected\t Weights\n")
        prediction_list = []
        actual_list = []
        for _ in range(self.threshold):
            for inputs, label in zip(training_inputs, labels):
                prediction = self.predict(inputs)
                self.weights[1:] += self.learning_rate * (label - prediction) * inputs
                self.weights[0] += self.learning_rate * (label - prediction)
                # print(str(label) +"\t" + str(prediction) +"\t" + str(self.weights))
                prediction_list.append(prediction)
                actual_list.append(label)
        
        accuracy = self.accuracy_metric(actual_list, prediction_list)
        print("Accuracy: ", accuracy)
        return prediction_list

    def accuracy_metric(self, actual, predicted):
        correct = 0
        for i in range(len(actual)):
            if actual[i] == predicted[i]:
                correct += 1
        return correct / float(len(actual)) * 100.0 