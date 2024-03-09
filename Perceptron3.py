"""
A simple implementation of a MLP using NumPy
"""

import os

from sklearn.model_selection import train_test_split

import numpy as np
# import matplotlib.pyplot as plt
import pandas as pd
from scipy.special import expit  # numerically stable sigmoid function
import pickle


def sigma(x):
    return expit(x)


def sigma_prime(x):
    u = sigma(x)
    return u * (1 - u)


def relu(x):
    return x * (x > 0)


def relu_prime(x):
    return (x > 0)


# Weight initialization
def kaiming(network_config, l):
    return np.random.normal(size=(network_config[l + 1], network_config[l])) * np.sqrt(2. / network_config[l])


# Multilayer Perceptron Class
def normalize(X):
    return 2. * (X / 255. - 0.5)

batch_size = 256
class NeuralNetwork(object):
    def __init__(self, n_input, hidden_layers, n_classes, learning_rate, hidden_layer_activation, output_activation):

        network_config = (n_input,) + hidden_layers + (n_classes,)

        self.test_data = None
        self.x_test = None
        self.y_test = None
        self.y_valid = None
        self.x = None
        self.y = None
        self.x_valid = None
        self.x_train = None
        self.y_train = None
        self.train_data = None
        self.n_layers = len(network_config)

        if hidden_layer_activation == 'relu':
            self.hidden_layer_activation = relu
        if hidden_layer_activation == 'sigma':
            self.hidden_layer_activation = sigma

        if output_activation == 'relu':
            self.output_activation = relu
        if output_activation == 'sigma':
            self.output_activation = sigma

        if hidden_layer_activation == 'relu':
            self.hidden_layer_activation_prime = relu_prime
        if hidden_layer_activation == 'sigma':
            self.hidden_layer_activation_prime = sigma_prime

        if output_activation == 'relu':
            self.output_activation_prime = relu_prime
        if output_activation == 'sigma':
            self.output_activation_prime = sigma_prime

        self.learning_rate = learning_rate

        # Weights
        self.W = [kaiming(network_config, l) for l in range(self.n_layers - 1)]
        # Bias
        self.b = [np.zeros((network_config[l], 1)) for l in range(1, self.n_layers)]

        # Pre-activation
        self.z = [None for l in range(1, self.n_layers)]
        # Activations
        self.a = [None for l in range(self.n_layers)]
        # Gradients
        self.dW = [None for l in range(self.n_layers - 1)]
        self.db = [None for l in range(1, self.n_layers)]

    def grouped_rand_idx(self, n_total, batch_size):
        idx = np.random.permutation(n_total)
        return [idx[i:i + batch_size] for i in range(0, len(idx), batch_size)]

    def optimize(self, epochs):
        self.prediction(self.x_valid, self.y_valid, 0, mode="valid")

        for epoch in range(epochs):
            self.train_one_iteration(self.x_train, self.y_train)
            self.prediction(self.x_valid, self.y_valid, epoch + 1, mode="valid")

    def train_one_iteration(self, train_x, train_y):
        eta = self.learning_rate / batch_size
        idx_list = self.grouped_rand_idx(len(train_x), batch_size)
        for idx in idx_list:
            # Get batch of random training samples
            x_batch, y_batch = train_x[idx], train_y[idx]
            self.feedforward(x_batch)
            self.backprop_gradient_descent(y_batch, eta)
    def backprop_gradient_descent(self, Y, eta):
        # Backpropagation
        delta = (self.a[-1] - Y) * self.output_activation_prime(self.z[self.n_layers - 2])
        self.dW[self.n_layers - 2] = np.matmul(delta.T, self.a[self.n_layers - 2])
        self.db[self.n_layers - 2] = np.sum(delta.T, axis=1, keepdims=True)

        for l in reversed(range(self.n_layers - 2)):
            delta = np.matmul(delta, self.W[l + 1]) * self.hidden_layer_activation_prime(self.z[l])
            self.dW[l] = np.matmul(self.a[l].T, delta).T
            self.db[l] = np.sum(delta.T, axis=1, keepdims=True)

        # Gradient descent: Update Weights and Biases
        for l in range(self.n_layers - 1):
            self.W[l] -= eta * self.dW[l]
            self.b[l] -= eta * self.db[l]

        # Reset gradients
        self.dW = [None for l in range(self.n_layers - 1)]
        self.db = [None for l in range(self.n_layers - 1)]

    def feedforward(self, X):
        self.a[0] = X
        for l in range(self.n_layers - 2):
            self.z[l] = np.matmul(self.a[l], self.W[l].T) + self.b[l].T  # Pre-activation hidden layer
            self.a[l + 1] = self.hidden_layer_activation(self.z[l])  # Activation hidden layer
        self.z[-1] = np.matmul(self.a[-2], self.W[-1].T) + self.b[-1].T  # Pre-activation output layer
        self.a[-1] = self.output_activation(self.z[-1])  # Activation output layer

    def pred(self, X, Y):
        neurons = X
        for l in range(self.n_layers - 2):
            neurons = self.hidden_layer_activation(np.matmul(neurons, self.W[l].T) + self.b[l].T)
        logits = np.matmul(neurons, self.W[-1].T) + self.b[-1].T
        accuracy = (np.argmax(logits, axis=1) == np.argmax(Y, axis=1)).sum() / len(X)
        loss = np.sum((Y - self.output_activation(logits)) ** 2) / len(X)
        return loss, accuracy

    def prediction(self, X, Y, epoch = None, mode = 'test'):
        loss, accuracy = self.pred(X, Y)
        print('epoch {1} {0}_loss {2:.6f} {0}_accuracy {3:.4f}'.format(mode, epoch, loss, accuracy), flush=True)

    def loadTrainDataFromCsv(self, path):
        self.train_data = pd.read_csv(path)
        self.x = self.train_data.iloc[:, 1:].values  # Нормализация значений пикселей
        self.y = self.train_data.iloc[:, 0].values.reshape(-1, 1)  # Целевая переменная для обучения
        self.x_train, self.x_valid, self.y_train, self.y_valid = train_test_split(self.x, self.y, test_size=0.05)
        self.x_train, self.x_valid, self.y_train, self.y_valid = normalize(self.x_train), normalize(
            self.x_valid), self.y_train, self.y_valid
        iden = np.eye(10)
        self.y_train = iden[self.y_train.flatten()]
        self.y_valid = iden[self.y_valid.flatten()]

    def loadTestDataFromCsv(self, path):
        self.test_data = pd.read_csv(path)
        self.x_test = self.test_data.iloc[:, 1:].values / 255.0  # Нормализация значений пикселей для теста
        self.y_test = self.test_data.iloc[:, 0].values.reshape(-1, 1)  # Целевая переменная для теста
        iden = np.eye(10)
        self.y_test = iden[self.y_test.flatten()]

    def save(self, filename):
        with open(filename, 'wb') as f:
            pickle.dump(self, f)

    @staticmethod
    def load(filename):
        with open(filename, 'rb') as f:
            return pickle.load(f)

    def test(self):
        self.prediction(self.x_test, self.y_test, mode="test")

    def dispose(self):
        self.test_data = None
        self.x_test = None
        self.y_test = None
        self.y_valid = None
        self.x = None
        self.y = None
        self.x_valid = None
        self.x_train = None
        self.y_train = None
        self.train_data = None


from tensorflow.python.client import device_lib

# def get_available_devices():
#     local_device_protos = device_lib.list_local_devices()
#     return [x.name for x in local_device_protos]
#
# print(get_available_devices())

learning_rate = 0.05
epochs = 15
n_input = 784
n_classes = 10
hidden_layers = (128,128,128)
# Network configuration


# Initialize network
# Самое мощное - релу на скрытые слои + сигмоид на аутпут. Везде сигмоид = медленно. Везде релу = не обучается, глупая нейросеть
network = NeuralNetwork(n_input, hidden_layers, n_classes, learning_rate, 'relu', 'sigma')

network.loadTrainDataFromCsv("mnist_train.csv")
network.loadTestDataFromCsv("mnist_test.csv")

# Start training
network.optimize(epochs)

network.test()

network.dispose()

network.save("asd.mlp")

network2 = NeuralNetwork.load("asd.mlp")

network2.loadTestDataFromCsv("mnist_test.csv")
# Compute test accuracy and loss
network2.test()
print('тестим на кастомном датасете')
network2.loadTestDataFromCsv("images_data2.csv")
# Compute test accuracy and loss
network2.test()
print('обучаем на кастомном датасете')
network2.loadTrainDataFromCsv("images_data2.csv")

network2.optimize(3)

print('тестим после обучения на кастомном датасете на mnist_test.csv')
network2.loadTestDataFromCsv("mnist_test.csv")

# Compute test accuracy and loss
network2.test()
