import cv2
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image

# Загрузка модели
model = load_model(r'C:\Dev\NEYRO_CATS\cat_classifier_model.keras')

# Открываем видеопоток (может потребоваться изменить индекс в случае использования веб-камеры)
cap = cv2.VideoCapture(0)

while True:
    # Захватываем кадр из видеопотока
    ret, frame = cap.read()

    # Преобразование изображения в формат, подходящий для модели
    img = cv2.resize(frame, (256, 256))  # изменение размера до 150x150
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array /= 255.0  # Нормализация значений пикселей к диапазону [0, 1]

    # Получение предсказания от модели
    prediction = model.predict(img_array)

    # Вывод предсказания в консоль
    print("Prediction:", prediction)

    # Проверка, является ли предсказание котом (предположим, что класс кота - 0)
    if prediction[0][0] > 0.5:
        print("Cat detected!")

        # Определение координат прямоугольника (предполагаем, что у вас есть функция для обнаружения объектов)
        x, y, w, h = 50, 50, 100, 100  # настройте эти значения по вашему усмотрению или используйте результат обнаружения

        # Рисование прямоугольника вокруг обнаруженного объекта (кота)
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

        # Отображение кадра с подписью
        cv2.putText(frame, 'Cat', (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    else:
        print("No cat detected.")

    # Отображение кадра
    cv2.imshow('Cat Classifier', frame)

    # Прерывание цикла по клавише 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Закрытие видеопотока и окна
cap.release()
cv2.destroyAllWindows()
