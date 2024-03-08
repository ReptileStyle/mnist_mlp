import pandas as pd
# from keras.models import Sequential
# from keras.layers import Dense, Flatten
# from keras.layers import Dropout
# from keras.preprocessing import image
# from google.colab import files
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image


import tensorflow as tf
from tensorflow.keras import layers, models
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
import pandas as pd
from tensorflow.keras.optimizers import Adam



classes = [1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28]

train_data = pd.read_csv('mnist_train.csv')
test_data = pd.read_csv('mnist_test.csv')
custom_data = pd.read_csv('images_data2.csv')
# Загрузка данных
x_train = train_data.iloc[:, 1:]
y_train = train_data.iloc[:, 0]
x_test = test_data.iloc[:, 1:]
y_test = test_data.iloc[:, 0]
x_custom = custom_data.iloc[:, 1:]
y_custom = custom_data.iloc[:, 0]

# Предварительная обработка данных
# Преобразование меток в one-hot encoding
encoder = LabelEncoder()
y_train_encoded = encoder.fit_transform(y_train.values.ravel())
y_test_encoded = encoder.transform(y_test.values.ravel())
y_custom_encoded = encoder.transform(y_custom.values.ravel())

# Перевод данных в формат, совместимый с TensorFlow
x_train = x_train.values.reshape(-1, 28, 28, 1)
x_test = x_test.values.reshape(-1, 28, 28, 1)
x_custom = x_custom.values.reshape(-1, 28, 28, 1)
y_train_encoded = tf.keras.utils.to_categorical(y_train_encoded)
y_test_encoded = tf.keras.utils.to_categorical(y_test_encoded)
y_custom_encoded = tf.keras.utils.to_categorical(y_custom_encoded)

# Создание модели CNN
model = models.Sequential([
    layers.Conv2D(16, (3, 3), activation='relu', input_shape=(28, 28, 1)),
    layers.MaxPooling2D((2, 2)),
    # layers.Conv2D(128, (3, 3), activation='relu'),
    # layers.MaxPooling2D((2, 2)),
    layers.Conv2D(30, (3, 3), activation='relu'),
    layers.Flatten(),
    layers.Dense(30, activation='relu'),
    layers.Dense(len(encoder.classes_), activation='softmax')  # количество классов равно количеству уникальных меток
])

# Задание learning rate
learning_rate = 0.00025  # например, можно выбрать значение 0.001

# Создание оптимизатора с заданным learning rate
optimizer = Adam(learning_rate=learning_rate)

# Компиляция модели
model.compile(optimizer=optimizer,
              loss='categorical_crossentropy',
              metrics=['accuracy'])

# Обучение модели
model.fit(x_train, y_train_encoded, epochs=5, batch_size=256, validation_data=(x_test, y_test_encoded))

# Оценка производительности модели на тестовом наборе
custom_test_loss, custom_test_accuracy = model.evaluate(x_custom, y_custom_encoded)

print("Test Accuracy:", custom_test_accuracy)
print("Test Loss:", custom_test_loss)


# # Загрузка данных тестового набора
# x_test = pd.read_csv('arabic_data/csvTestImages 3360x1024.csv', header=None).values
# y_test = pd.read_csv('arabic_data/csvTestLabel 3360x1.csv', header=None).values.reshape(-1, 1)
#
# # Загрузка данных тренировочного набора
# x_train = pd.read_csv('arabic_data/csvTrainImages 13440x1024.csv', header=None).values
# y_train = pd.read_csv('arabic_data/csvTrainLabel 13440x1.csv', header=None).values.reshape(-1, 1)
#
# x_train = x_train.astype('float32')
# x_test = x_test.astype('float32')
# x_train /= 255
# x_test /= 255
#
# # Преобразуем метки классов в удобный для обучения нейронной сети формат (one hot encoding)
# iden = np.eye(28)
# y_train = iden[y_train.flatten() - 1]
# y_test = iden[y_test.flatten() - 1]

