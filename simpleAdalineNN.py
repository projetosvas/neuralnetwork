## Simple Layer Adaline Network - Regression Problem
# Created by: Vinicius Amorim Santos 
# Date: 27/03/2025
# Version: 1.0.0

import numpy as np
import matplotlib.pyplot as plt

class Adaline():
    def __init__(self, inputSize, learningRate, epochs):
        self.inputSize = inputSize
        self.lr = learningRate
        self.epochs = epochs
        self.weights = np.random.random(inputSize)
        self.bias = np.random.random(1)
        self.error = 0

    def fit(self, X, y):
        for i in range(self.epochs):
            print(f'Weights: {self.weights} Bias: {self.bias} error: {self.error} epoch: {i}')
            for xi, target in zip(X,y):
                output = np.dot(xi, self.weights) + self.bias
                self.error = ((target - output)**2)/np.size(X)
                self.weights = self.weights + self.lr * self.error * xi
                self.bias = self.bias + self.lr * self.error


X = np.array([1, 2, 3, 4, 5])
y = np.array([3, 5, 7, 9, 11])

print('Treino Adaline: ')
adaline = Adaline(1, 0.1, 10)
adaline.fit(X,y)

plt.plot(X, X * adaline.weights + adaline.bias)
plt.scatter(X, y, color='r')
plt.show()