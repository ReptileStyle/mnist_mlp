import tensorflow as tf
from keras.applications import VGG16
from matplotlib import pyplot as plt
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten, BatchNormalization
from tensorflow.keras import layers, models
# Параметры для обучения
batch_size = 64
num_classes = 15

# Загрузка предобученной модели VGG16 без полносвязных слоев
base_model = VGG16(weights='imagenet', include_top=False, input_shape=(224, 224, 3), classes=1000)

# Замораживаем веса базовой модели
base_model.trainable = False


from tensorflow.keras.preprocessing.image import ImageDataGenerator
with tf.device('/GPU:0'):
    # Подготовка данных

    # train_datagen = ImageDataGenerator(rescale=1./255)
    train_datagen = ImageDataGenerator( # для аугментации
        rescale=1./255,
        rotation_range=20,
        width_shift_range=0.2,
        height_shift_range=0.2,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True,
    )

    train_generator = train_datagen.flow_from_directory(
        'vegetableImages/train',
        target_size=(224, 224),
        batch_size=batch_size,
        class_mode='categorical',
        subset='training'  # указание на использование обучающего набора
    )

    # Создание генератора изображений для тестовых данных
    test_datagen = ImageDataGenerator(rescale=1./255)
    test_generator = test_datagen.flow_from_directory(
        'VegetableImages/test',
        target_size=(224, 224),
        batch_size=batch_size,
        class_mode='categorical')

    # Создание генератора изображений для валидационных данных
    val_datagen = ImageDataGenerator(rescale=1./255)
    val_generator = val_datagen.flow_from_directory(
        'VegetableImages/validation',
        target_size=(224, 224),
        batch_size=batch_size,
        class_mode='categorical')


    print('num classes = ' + str(train_generator.num_classes))
    # Добавляем полносвязные слои поверх базовой модели
    model = models.Sequential([
        base_model,
        layers.Flatten(),
        layers.Dense(256, activation='relu'),
        layers.Dropout(0.5),
        layers.Dense(train_generator.num_classes, activation='softmax')
    ])

    # Компиляция модели
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

    # Вывод информации о модели
    model.summary()


    # Обратные вызовы для сохранения модели и истории обучения
    checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
        filepath='vegetable_classification_model_aug_2.h5',
        monitor='val_accuracy',
        save_best_only=True,
        verbose=1)

    # Обучение модели
    history = model.fit(
        train_generator,
        epochs=13,
        validation_data=test_generator,
        callbacks = checkpoint_callback
    )

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




