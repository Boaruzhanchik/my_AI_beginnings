import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from PIL import Image
import os
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import matplotlib.pyplot as plt
import numpy as np


def is_image_file(file_path):
    try:
        image = Image.open(file_path)
        image.verify()
        return True
    except Exception as e:
        print(f"Invalid image file: {file_path}")
        return False


def generator_with_image_check(generator, is_training=True):
    while True:
        try:
            batch_x, batch_y = next(generator)
        except StopIteration:
            generator.on_epoch_end()
            continue

        valid_indices = [i for i, file_path in enumerate(generator.filepaths) if is_image_file(file_path)]

        # Проверка, что есть хотя бы одно изображение в батче
        if not valid_indices:
            print("No valid images in the batch. Skipping batch.")
            continue

        batch_x = np.array([batch_x[i] for i in valid_indices])
        batch_y = np.array([batch_y[i] for i in valid_indices])

        yield batch_x, batch_y


# Задайте путь к вашим данным (изображениям котов и собак)
train_data_dir = r'C:\Dev\NEYRO_CATS\train_dataset'
validation_data_dir = r'C:\Dev\NEYRO_CATS\test_dataset'
# Задайте параметры обучения
batch_size = 32
epochs = 10
img_height = 150
img_width = 150

# Создайте генераторы для тренировочных и валидационных данных
train_datagen = ImageDataGenerator(rescale=1./255)
validation_datagen = ImageDataGenerator(rescale=1./255)


train_generator = train_datagen.flow_from_directory(
    train_data_dir,
    target_size=(img_height, img_width),
    batch_size=batch_size,
    class_mode='binary'
)

validation_generator = validation_datagen.flow_from_directory(
    validation_data_dir,
    target_size=(img_height, img_width),
    batch_size=batch_size,
    class_mode='binary'
)

# Создайте модель нейронной сети
model = models.Sequential([
    layers.Conv2D(32, (3, 3), activation='relu', input_shape=(img_height, img_width, 3)),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(128, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    layers.Flatten(),
    layers.Dense(128, activation='relu'),
    layers.Dense(1, activation='sigmoid')
])

# Скомпилируйте модель
model.compile(optimizer='adam',
              loss='binary_crossentropy',
              metrics=['accuracy'])

# Обучите модель
history = model.fit(
    generator_with_image_check(train_generator),
    steps_per_epoch=train_generator.samples // batch_size,
    epochs=epochs,
    validation_data=generator_with_image_check(validation_generator, is_training=False),
    validation_steps=validation_generator.samples // batch_size
)

# Отобразите графики точности и потерь
acc = history.history['accuracy']
val_acc = history.history['val_accuracy']
loss = history.history['loss']
val_loss = history.history['val_loss']

epochs_range = range(epochs)

plt.figure(figsize=(8, 8))
plt.subplot(1, 2, 1)
plt.plot(epochs_range, acc, label='Training Accuracy')
plt.plot(epochs_range, val_acc, label='Validation Accuracy')
plt.legend(loc='lower right')
plt.title('Training and Validation Accuracy')

plt.subplot(1, 2, 2)
plt.plot(epochs_range, loss, label='Training Loss')
plt.plot(epochs_range, val_loss, label='Validation Loss')
plt.legend(loc='upper right')
plt.title('Training and Validation Loss')

plt.show()
