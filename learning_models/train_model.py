from tensorflow.keras import layers, models, regularizers
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import matplotlib.pyplot as plt
import os
import scipy
from tensorflow.keras.callbacks import EarlyStopping

# Задайте пути к обучающим и тестовым данным
train_data_dir = os.path.abspath(r'C:\Dev\NEYRO_CATS\train_dataset')
test_data_dir = os.path.abspath(r'C:\Dev\NEYRO_CATS\test_dataset')

# Создайте генераторы данных
train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest'
)
test_datagen = ImageDataGenerator(rescale=1./255)

train_generator = train_datagen.flow_from_directory(
    train_data_dir,
    target_size=(256, 256),
    batch_size=64,
    class_mode='binary'
)

test_generator = test_datagen.flow_from_directory(
    test_data_dir,
    target_size=(256, 256),
    batch_size=64,
    class_mode='binary'
)

# Создайте модель нейронной сети
dropout_rate = 0.3

model = models.Sequential()
model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=(256, 256, 3), kernel_regularizer=regularizers.l2(0.0001)))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(64, (3, 3), activation='relu', kernel_regularizer=regularizers.l2(0.0001)))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(128, (3, 3), activation='relu', kernel_regularizer=regularizers.l2(0.0001)))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Flatten())
model.add(layers.Dropout(dropout_rate))
model.add(layers.Dense(64, activation='relu', kernel_regularizer=regularizers.l2(0.0001)))
model.add(layers.Dense(1, activation='sigmoid'))

# Скомпилируйте модель
model.compile(optimizer='adam',
              loss='binary_crossentropy',
              metrics=['accuracy'])

# Обучение модели
early_stopping = EarlyStopping(monitor='val_loss', patience=3)

history = model.fit(
    train_generator,
    epochs=5,
    validation_data=test_generator,
    callbacks=[early_stopping]
)
# Сохранение модели в формате Keras
model.save('cat_classifier_model.keras')

# Построение графика точности
plt.plot(history.history['accuracy'], label='Точность на обучающем наборе')
plt.plot(history.history['val_accuracy'], label='Точность на тестовом наборе')
plt.xlabel('Эпоха')
plt.ylabel('Точность')
plt.legend()
plt.show()

# Построение графика потерь
plt.plot(history.history['loss'], label='Потери на обучающем наборе')
plt.plot(history.history['val_loss'], label='Потери на тестовом наборе')
plt.xlabel('Эпоха')
plt.ylabel('Потери')
plt.legend()
plt.show()

