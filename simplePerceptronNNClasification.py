## Simple Layer Perceptron Network - Classification Problem
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

    def step(self, X):
        if X >= 0:
            return int(1)
        else:
            return int(0)

    def fit(self, X, y):
        for i in range(self.epochs):
            print(f'Weight: {self.weights} Bias: {self.bias} epoch: {i}')
            for xi, target in zip(X, y):
                output = self.step(np.dot(xi, self.weights) - self.bias)
                self.error = (target - output)
                print(f'y: {target} output: {output} Error: {self.error}')
                self.weights = self.weights + self.lr * (self.error) * xi
                self.bias = self.bias + self.lr * self.error


    def predict(self,X):
        Predict = np.dot(X, self.weights) + self.bias
        
        return Predict


# Data

X = np.array([[4, 1],
              [5, 4],
              [3, 1],
              [-3, 1],
              [0, -2],
              [1, 0]])

y = np.array([1, 1, 1, 0, 0, 0])

# Train

print('Treino Perceptron:')
perceptron = Perceptron(2, 0.1, 10)
perceptron.fit(X, y) 


x1 = np.array([-5, -4, -3, -2, -1, 0, 1, 2, 3, 4, 5])
x2 = -(perceptron.weights[0]/perceptron.weights[1])*x1 - perceptron.bias

fig, ax = plt.subplot()
ax.scatter(X[:3,0], X[:3,1], color='r')
ax.scatter(X[3:,0], X[3:,1], color='b')
ax.plot(x1, x2)
ax.xlim(-6, 6)
ax.ylim(-2, 6)
plt.show()