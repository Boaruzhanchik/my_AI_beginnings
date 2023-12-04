from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import matplotlib.pyplot as plt

# Установите путь к вашим данным
train_data_dir = r'C:\Dev\NEYRO_CATS\train_dataset'
test_data_dir = r'C:\Dev\NEYRO_CATS\test_dataset'

# Определите параметры модели
input_shape = (256, 256, 3)
num_classes = 2
image_width = 256
image_height = 256
batch_size = 32

# Определите генераторы данных для обучения и тестирования
train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=15,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest',
    brightness_range=[0.9, 1.1],
)
test_datagen = ImageDataGenerator(rescale=1./255)

train_generator = train_datagen.flow_from_directory(
    train_data_dir,
    target_size=(image_width, image_height),
    batch_size=batch_size,
    class_mode='categorical',
    classes=['cats', 'dogs']
)

test_generator = test_datagen.flow_from_directory(
    test_data_dir,
    target_size=(image_width, image_height),
    batch_size=batch_size,
    class_mode='categorical',
    classes=['cats', 'dogs']
)

# Создайте модель
model = Sequential()
model.add(Conv2D(32, (3, 3), activation='relu', input_shape=input_shape))
model.add(MaxPooling2D((2, 2)))
model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(MaxPooling2D((2, 2)))
model.add(Conv2D(128, (3, 3), activation='relu'))
model.add(MaxPooling2D((2, 2)))
model.add(Flatten())
model.add(Dense(128, activation='relu'))
model.add(Dense(num_classes, activation='softmax'))

# Скомпилируйте модель
model.compile(optimizer=Adam(learning_rate=0.001), loss='categorical_crossentropy', metrics=['accuracy'])

# Обучите модель
history = model.fit(
    train_generator,
    steps_per_epoch=len(train_generator),
    epochs=7,
    validation_data=test_generator,
    validation_steps=len(test_generator)
)

# Сохраните модель
model.save('cat_classifier_models.keras')

# Отобразите графики обучения
plt.plot(history.history['accuracy'], label='Точность на обучении')
plt.plot(history.history['val_accuracy'], label='Точность на валидации')
plt.title('Точность модели')
plt.xlabel('Эпоха')
plt.ylabel('Точность')
plt.legend()
plt.show()

plt.plot(history.history['loss'], label='Потери на обучении')
plt.plot(history.history['val_loss'], label='Потери на валидации')
plt.title('Функция потерь модели')
plt.xlabel('Эпоха')
plt.ylabel('Потери')
plt.legend()
plt.show()
