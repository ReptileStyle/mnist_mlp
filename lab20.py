import tensorflow as tf
from matplotlib import pyplot as plt
from tensorflow.keras import layers, models
from sklearn.preprocessing import LabelEncoder
import pandas as pd
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import os

from tensorflow.python.client import device_lib

# def get_available_devices():
#     local_device_protos = device_lib.list_local_devices()
#     return [x.name for x in local_device_protos]
#

gpus = tf.config.list_physical_devices('GPU')
if gpus:
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)
        print('set memory growth ' + str(gpu))

with tf.device('/GPU:0'):
    train_datagen = ImageDataGenerator(rescale=1./255)

    #     # Подготовка данных
    # train_datagen = ImageDataGenerator( // для аугментации
    #     rescale=1./255,
    #     rotation_range=20,
    #     width_shift_range=0.2,
    #     height_shift_range=0.2,
    #     shear_range=0.2,
    #     zoom_range=0.2,
    #     horizontal_flip=True,
    # )
    train_generator = train_datagen.flow_from_directory(
        'VegetableImages/train',
        target_size=(224, 224),
        batch_size=128,
        class_mode='categorical')

    # Создание генератора изображений для тестовых данных
    test_datagen = ImageDataGenerator(rescale=1./255)
    test_generator = test_datagen.flow_from_directory(
        'VegetableImages/test',
        target_size=(224, 224),
        batch_size=12,
        class_mode='categorical')

    # Создание генератора изображений для валидационных данных
    val_datagen = ImageDataGenerator(rescale=1./255)
    val_generator = val_datagen.flow_from_directory(
        'VegetableImages/validation',
        target_size=(224, 224),
        batch_size=128,
        class_mode='categorical')

    # Создание модели CNN
    model = models.Sequential([
        layers.Conv2D(16, (3, 3), activation='relu', input_shape=(224, 224, 3)),
        layers.MaxPooling2D((2, 2)),
        layers.Conv2D(30, (3, 3), activation='relu'),
        layers.Flatten(),
        layers.Dense(30, activation='relu'),
        layers.Dense(train_generator.num_classes, activation='softmax')
    ])

    # Задание learning rate
    learning_rate = 0.0025

    # Создание оптимизатора с заданным learning rate
    optimizer = Adam(learning_rate=learning_rate)

    # Компиляция модели
    model.compile(optimizer=optimizer,
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])

    # Обратные вызовы для сохранения модели и истории обучения
    checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
        filepath='vegetable_classification_model.h5',
        monitor='val_accuracy',
        save_best_only=True,
        verbose=1)

    # Обучение модели
    history = model.fit(train_generator, epochs=13, validation_data=test_generator, callbacks = checkpoint_callback)

    # Оценка производительности модели на валидационном наборе
    val_loss, val_accuracy = model.evaluate(val_generator)

    print("Validation Accuracy:", val_accuracy)
    print("Validation Loss:", val_loss)

    # Вывод кривых точности
    plt.plot(history.history['accuracy'], label='accuracy')
    plt.plot(history.history['val_accuracy'], label='val_accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend(loc='lower right')
    plt.show()

    # Сохранение модели
    model.save('vegetable_classification_model.h5')
