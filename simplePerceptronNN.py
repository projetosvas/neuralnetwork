## Simple Layer Perceptron Network - Regression Problem
# Created by: Vinicius Amorim Santos 
# Date: 27/03/2025
# Version: 1.0.0

import numpy as np
import matplotlib.pyplot as plt

class Perceptron():
    def __init__(self, inputSize, learningRate, epochs):
        self.inputSize = inputSize # Size of Input Vector (Number of Features)
        self.lr = learningRate # Learning Rate
        self.epochs = epochs # Number of learning cicles
        self.weights = np.random.random(self.inputSize) # Weights Vector
        self.bias = np.random.random(1) # Bias
        self.error = 0

    def fit(self, X, y):
        for i in range(self.epochs):
            print(f'Weight: {self.weights} Bias: {self.bias} Error: {self.error} epoch: {i}')
            for xi, target in zip(X, y):
                output = np.dot(xi, self.weights) + self.bias
                self.error = (target - output)
                self.weights = self.weights + self.lr * (self.error) * xi
                self.bias = self.bias + self.lr * self.error


    def predict(self,X):
        
        Predict = np.dot(X, self.weights) + self.bias
        
        return Predict


# Data

X = np.array([1, 2, 3, 4, 5])
y = np.array([3, 5, 7, 9, 11])

# Train

perceptron = Perceptron(1, 0.1, 20)
perceptron.fit(X, y) 

print(perceptron.predict(7))

plt.plot(X, X * perceptron.weights + perceptron.bias)
plt.scatter(X, y, color='r')
plt.show()