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
        if X > 0:
            return int(1)
        else:
            return int(0)

    def fit(self, X, y):
        for i in range(self.epochs):
            print(f'Weight: {self.weights} Bias: {self.bias} epoch: {i}')
            for xi, target in zip(X, y):
                output = self.step(np.dot(xi, self.weights) + self.bias)
                self.error = (target - output)
                self.weights = self.weights + self.lr * (self.error) * xi
                self.bias = self.bias + self.lr * self.error


    def predict(self,X):
        Predict = np.dot(X, self.weights) + self.bias
        
        return Predict


# Data

X = np.array([[3,2],
              [1,2],
              [4,3],
              [0,2],
              [2,0],
              [1,1],
              [2,4],
              [1,5],
              [5,1],
              [4,1],
              [0,4],
              [3,0]])

y = np.array([1, 0, 1, 0, 0, 0, 1, 1, 1, 1, 0, 0])

# Train

print('Treino Perceptron:')
perceptron = Perceptron(2, 0.1, 10)
perceptron.fit(X, y) 


x1 = np.linspace(-2, 6, 100)  
x2 = -(perceptron.weights[0] / perceptron.weights[1]) * x1 - (perceptron.bias / perceptron.weights[1])

colors = np.array(['red', 'blue'])

plt.scatter(X[:, 0], X[:, 1], c=colors[y], label='Pontos')
plt.plot(x1, x2)
plt.grid(True)
plt.show()