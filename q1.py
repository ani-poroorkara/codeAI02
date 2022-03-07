"""
Question 1: Part A, B, C
Data Point Creation
"""
import numpy as np
import pandas as pd
import matplotlib as plt

from resources.adaline import AdalineSGD, AdaptiveLinearNeuron, SigmoidNeuron

X1 = np.random.normal(0,1,5000)
X2 = X1**2
X3 = X1**3
eps = np.array(np.random.normal(0,0.25,5000))
y = -1+(0.5*X1)-(2*(X1**2))+(0.3*(X1**3))+eps

X1 = pd.DataFrame(X1, columns = ['X1'])
X2 = pd.DataFrame(X2, columns = ['X2'])
X3 = pd.DataFrame(X3, columns = ['X3'])

result = pd.concat([X1, X2, X3], axis = 1)

df = pd.DataFrame(result, columns = ['X1', 'X2', 'X3'])
dfp = np.array(df)
x1 = np.array(X1)

"""
Question 2: Part D
Adaline with
    a. Batch gradient descent
    b. Stochastic gradient descent
"""

aln = AdaptiveLinearNeuron()
aln.fit(x1, y)
plt.plot(range(1, len(aln.cost) + 1), np.log10(aln.cost), marker='o')
plt.title("Adaline: Batch Gradient Descent")
plt.xlabel("Epochs")
plt.ylabel("MSE")

plt.show()

aln = AdaptiveLinearNeuron(rate=0.0001)
aln.fit(x1, y)
plt.plot(range(1, len(aln.cost) + 1), np.log10(aln.cost), marker='o')
plt.title("Adaline: Batch Gradient Descent")
plt.xlabel("Epochs")
plt.ylabel("MSE")

plt.show()

aln = AdaptiveLinearNeuron(rate=0.00001)
aln.fit(x1, y)
plt.plot(range(1, len(aln.cost) + 1), np.log10(aln.cost), marker='o')
plt.title("Adaline: Batch Gradient Descent")
plt.xlabel("Epochs")
plt.ylabel("MSE")

plt.show()

sgd = AdalineSGD(rate=0.01, niter=10)
sgd.fit(x1, y)
plt.plot(range(1, len(sgd.cost) + 1), sgd.cost, marker='o')
plt.title("Adaline: Stochastic Gradient Descent")
plt.xlabel('Epochs')
plt.ylabel('Average Cost')
plt.show()
plt.show()

sgd = AdalineSGD(rate=0.001, niter=10)
sgd.fit(x1, y)
plt.plot(range(1, len(sgd.cost) + 1), sgd.cost, marker='o')
plt.title("Adaline: Stochastic Gradient Descent")
plt.xlabel('Epochs')
plt.ylabel('Average Cost')
plt.show()
plt.show()

sgd = AdalineSGD(rate=0.0001, niter=10)
sgd.fit(x1, y)
plt.plot(range(1, len(sgd.cost) + 1), sgd.cost, marker='o')
plt.title("Adaline: Stochastic Gradient Descent")
plt.xlabel('Epochs')
plt.ylabel('Average Cost')
plt.show()
plt.show()

sgd = AdalineSGD(rate=0.00001, niter=10)
sgd.fit(x1, y)
plt.plot(range(1, len(sgd.cost) + 1), sgd.cost, marker='o')
plt.title("Adaline: Stochastic Gradient Descent")
plt.xlabel('Epochs')
plt.ylabel('Average Cost')
plt.show()
plt.show()

"""
Question 2: Part E
Adaline with sigmoid activation function
"""
sig = SigmoidNeuron(rate=0.1)
sig.fit(x1, y)
plt.plot(range(1, len(sig.cost) + 1), sig.cost, marker='o')
plt.title("Adaline: Batch Gradient Descent with Sigmoid activation")
plt.xlabel("Epochs")
plt.ylabel("MSE")

plt.show()

sig = SigmoidNeuron(rate=0.001)
sig.fit(x1, y)
plt.plot(range(1, len(sig.cost) + 1), sig.cost, marker='o')
plt.title("Adaline: Batch Gradient Descent with Sigmoid activation")
plt.xlabel("Epochs")
plt.ylabel("MSE")

plt.show()

sig = SigmoidNeuron(rate=0.0001)
sig.fit(x1, y)
plt.plot(range(1, len(sig.cost) + 1), sig.cost, marker='o')
plt.title("Adaline: Batch Gradient Descent with Sigmoid activation")
plt.xlabel("Epochs")
plt.ylabel("MSE")

plt.show()

"""
Question 2: F
"""

#Degree = 1
x1 = np.array(X1)

aln = AdaptiveLinearNeuron(rate=0.01, niter=10)
aln.fit(x1, y)
aln.get_weights()

sgd = AdalineSGD(rate=0.01, niter=10)
sgd.fit(x1, y)
sgd.get_weights()

sig = SigmoidNeuron(rate=0.01, niter=10)
sig.fit(x1, y)
sig.get_weights()

#Degree = 2
result_2 = pd.concat([X1, X2], axis = 1)
x2 = pd.DataFrame(result_2, columns = ['X1', 'X2'])
x2 = np.array(x2)

aln = AdaptiveLinearNeuron(rate=0.01, niter=10)
aln.fit(x2, y)
aln.get_weights()

sgd = AdalineSGD(rate=0.01, niter=10)
sgd.fit(x2, y)
sgd.get_weights()

sig = SigmoidNeuron(rate=0.01, niter=10)
sig.fit(x2, y)
sig.get_weights()


#Degree = 3
result_f = pd.concat([X1, X2, X3], axis = 1)
df = pd.DataFrame(result_f, columns = ['X1', 'X2', 'X3'])
final = np.array(df)

aln = AdaptiveLinearNeuron(rate=0.01, niter=10)
aln.fit(final, y)
aln.get_weights()

sgd = AdalineSGD(rate=0.01, niter=10)
sgd.fit(final, y)
sgd.get_weights()

sig = SigmoidNeuron(rate=0.01, niter=10)
sig.fit(final, y)
sig.get_weights()