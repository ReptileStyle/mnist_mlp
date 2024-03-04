import numpy as np
import pickle
import pandas as pd


class MLP:
    def __init__(self, input_size, hidden_layers, output_size, learning_rate=0.1, activation_function='sigmoid'):
        self.input_size = input_size
        self.hidden_layers = hidden_layers
        self.output_size = output_size
        self.learning_rate = learning_rate

        if activation_function == 'sigmoid':
            self.activation_function = self.sigmoid
            self.activation_derivative = self.sigmoid_derivative
        elif activation_function == 'relu':
            self.activation_function = self.relu
            self.activation_derivative = self.relu_derivative

        self.weights = []
        self.biases = []
        layer_sizes = [input_size] + hidden_layers + [output_size]
        for i in range(len(layer_sizes) - 1):
            self.weights.append(np.random.randn(layer_sizes[i], layer_sizes[i + 1]))
            self.biases.append(np.random.randn(layer_sizes[i + 1]))

    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))

    def sigmoid_derivative(self, x):
        return x * (1 - x)

    def relu(self, x):
        return np.maximum(0, x)

    def relu_derivative(self, x):
        return np.where(x > 0, 1, 0)

    def save(self, filename):
        with open(filename, 'wb') as f:
            pickle.dump(self, f)

    @staticmethod
    def load(filename):
        with open(filename, 'rb') as f:
            return pickle.load(f)

    def forward_propagation(self, x):
        activations = [x]
        for i in range(len(self.hidden_layers)):
            x = self.activation_function(np.dot(x, self.weights[i]) + self.biases[i])
            activations.append(x)
        output = np.dot(x, self.weights[-1]) + self.biases[-1]
        activations.append(output)
        return activations

    def back_propagation(self, x, y, activations):
        errors = [None] * len(self.weights)
        errors[-1] = y - activations[-1]
        for i in range(len(self.weights) - 2, -1, -1):
            errors[i] = np.dot(errors[i + 1], self.weights[i + 1].T) * self.activation_derivative(activations[i + 1])
        for i in range(len(self.weights)):
            self.weights[i] += self.learning_rate * np.dot(activations[i].T, errors[i])
            self.biases[i] += self.learning_rate * np.sum(errors[i], axis=0)

    def train_single_instance(self, x, y):
        x = np.array(x, ndmin=2)
        y = np.array(y, ndmin=2)
        activations = self.forward_propagation(x)
        self.back_propagation(x, y, activations)

    def train(self, X, Y, epochs):
        data_size = len(X)
        for epoch in range(epochs):
            for i, (x, y) in enumerate(zip(X, Y), 1):
                self.train_single_instance(x, y)
                if i % (data_size // 20) == 0 or i == data_size:
                    percent_processed = i / data_size * 100
                    print(f"Epoch {epoch + 1}/{epochs}, {percent_processed:.1f}% of dataset processed")

    def predict(self, x):
        x = np.array(x, ndmin=2)
        return self.forward_propagation(x)[-1]


from sklearn.model_selection import ParameterGrid
from sklearn.preprocessing import MinMaxScaler

# Создание объекта для масштабирования
scaler = MinMaxScaler()

# Загрузка данных
train_data = pd.read_csv("mnist_train.csv")
test_data = pd.read_csv("mnist_test.csv")

# Разделение данных на входные и целевые переменные
X_train = train_data.iloc[:, 1:].values.reshape(-1, 784) / 255.0  # Нормализация значений пикселей
Y_train = train_data.iloc[:, 0].values.reshape(-1, 1)  # Целевая переменная для обучения

print(len(X_train))

X_test = test_data.iloc[:, 1:].values.reshape(-1, 784) / 255.0  # Нормализация значений пикселей для теста
Y_test = test_data.iloc[:, 0].values.reshape(-1, 1)  # Целевая переменная для теста


# Определение параметров для поиска по сетке
param_grid = {
    'hidden_layers': [[100, 50]], ##/ [256, 128], [64, 32], [32, 16]],
    'learning_rate': [0.1],
    'activation_function': ['relu']
}

best_accuracy = 0
best_params = None
best_mlp = None

# Поиск лучших параметров
for params in ParameterGrid(param_grid):
    mlp = MLP(input_size=784, output_size=10, **params)
    mlp.train(X_train, Y_train, epochs=1)
    predictions = mlp.predict(X_test)
    correct = sum(np.argmax(predictions[i]) == Y_test[i] for i in range(len(predictions)))
    accuracy = correct / len(predictions)
    print("Parameters:", params)
    print("Accuracy:", accuracy)
    if accuracy > best_accuracy:
        best_accuracy = accuracy
        best_params = params
        best_mlp = mlp

print("Best parameters:", best_params)
print("Best accuracy:", best_accuracy)

# Сохранение лучшей модели
best_mlp.save("best_mlp_model.pkl")
