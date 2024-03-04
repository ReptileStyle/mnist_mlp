"""
A simple implementation of a MLP using NumPy
"""

import os
import data2
import numpy as np

from scipy.special import expit # numerically stable sigmoid function

# Activation functions
def sigma(x):
    return expit(x)

def sigma_prime(x):
    u = sigma(x)
    return u*(1-u)

def relu(x):
    return x*(x>0)

def relu_prime(x):
    return (x>0)

# Weight initialization
def kaiming(network_config, l):
    return np.random.normal(size=(network_config[l+1], network_config[l])) * np.sqrt(2./network_config[l])

# Multilayer Perceptron Class
class NeuralNetwork(object):

    def __init__(self, network_config):

        self.n_layers = len(network_config)

        # Weights
        self.W = [kaiming(network_config, l) for l in range(self.n_layers-1)]
        # Bias
        self.b = [np.zeros((network_config[l], 1)) for l in range(1, self.n_layers)]

        # Pre-activation
        self.z = [None for l in range(1, self.n_layers)]
        # Activations
        self.a = [None for l in range(self.n_layers)]
        # Gradients
        self.dW = [None for l in range(self.n_layers-1)]
        self.db = [None for l in range(1, self.n_layers)]

    def grouped_rand_idx(self, n_total, batch_size):
        idx = np.random.permutation(n_total)
        return [idx[i:i+batch_size] for i in range(0, len(idx), batch_size)]

    def optimize(self, x_train, y_train, x_valid, y_valid, x_test, y_test, epochs, batch_size, learning_rate):

        eta = learning_rate / batch_size
        for epoch in range(epochs):

            if epoch % 1 == 0:
                self.prediction(x_valid, y_valid, epoch, mode="valid")

            idx_list = self.grouped_rand_idx(len(x_train), batch_size)
            for idx in idx_list:
                # Get batch of random training samples
                x_batch, y_batch = x_train[idx], y_train[idx]
                self.feedforward(x_batch)
                self.backprop_gradient_descent(y_batch, eta)

        self.prediction(x_valid, y_valid, epoch+1, mode="valid")
        # Compute test accuracy and loss
        self.prediction(x_test, y_test, epoch+1, mode="test")

    def backprop_gradient_descent(self, Y, eta):
        # Backpropagation
        delta = (self.a[-1] - Y) * sigma_prime(self.z[self.n_layers-2])
        self.dW[self.n_layers-2] = np.matmul(delta.T, self.a[self.n_layers-2])
        self.db[self.n_layers-2] = np.sum(delta.T, axis=1, keepdims=True)

        for l in reversed(range(self.n_layers-2)):
            delta = np.matmul(delta, self.W[l+1]) * relu_prime(self.z[l])
            self.dW[l] = np.matmul(self.a[l].T, delta).T
            self.db[l] = np.sum(delta.T, axis=1, keepdims=True)

        # Gradient descent: Update Weights and Biases
        for l in range(self.n_layers-1):
            self.W[l] -= eta * self.dW[l]
            self.b[l] -= eta * self.db[l]

        # Reset gradients
        self.dW = [None for l in range(self.n_layers-1)]
        self.db = [None for l in range(self.n_layers-1)]

    def feedforward(self, X):
        self.a[0] = X
        for l in range(self.n_layers-2):
            self.z[l] = np.matmul(self.a[l], self.W[l].T) + self.b[l].T     # Pre-activation hidden layer
            self.a[l+1] = relu(self.z[l])                                   # Activation hidden layer
        self.z[-1] = np.matmul(self.a[-2], self.W[-1].T) + self.b[-1].T     # Pre-activation output layer
        self.a[-1] = sigma(self.z[-1])                                      # Activation output layer

    def pred(self, X, Y):
        neurons = X
        for l in range(self.n_layers-2):
            neurons = relu(np.matmul(neurons, self.W[l].T) + self.b[l].T)
        logits = np.matmul(neurons, self.W[-1].T) + self.b[-1].T
        print(logits)
        accuracy = (np.argmax(logits, axis=1) == np.argmax(Y, axis=1)).sum() / len(X)
        loss = np.sum((Y - sigma(logits))**2) / len(X)
        return loss, accuracy

    def prediction(self, X, Y, epoch, mode):
        loss, accuracy = self.pred(X, Y)
        print('epoch {1} {0}_loss {2:.6f} {0}_accuracy {3:.4f}'.format(mode, epoch, loss, accuracy), flush=True)


# Load data
x_train, y_train, x_valid, y_valid, x_test, y_test, n_classes, n_input = data2.get_data(dataset="mnist", norm=True, one_hot=True)

print(x_train.shape)
print(y_train)

# np.savetxt('4.csv', x_train[:100], delimiter=',')

# Training parameters
learning_rate = 0.2
batch_size = 256
epochs = 1000

# Network configuration
network_config = (n_input,) + 3*(128,) + (n_classes,)

print(n_classes)
print(n_input)

# Initialize network
network = NeuralNetwork(network_config)



# Start training
network.optimize(x_train, y_train, x_valid, y_valid, x_test, y_test, epochs, batch_size, learning_rate)
