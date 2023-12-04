from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.preprocessing import image
import os

# Задайте путь к папке с изображениями
input_image_folder = r'C:\Dev\NEYRO_CATS\class_2_normalsize\cats_hide'
output_image_folder = r'C:\Dev\NEYRO_CATS\class_2_normalsize\cats_hide'

# Создайте объект ImageDataGenerator с параметрами аугментации
datagen = ImageDataGenerator(
    rotation_range=40,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest'
)

# Получите список файлов в папке с изображениями
images = [os.path.join(input_image_folder, img) for img in os.listdir(input_image_folder) if img.endswith(".jpg")]

# Создайте выходную папку, если её нет
os.makedirs(output_image_folder, exist_ok=True)

# Проход по изображениям и сохранение аугментированных версий в другую папку
for img_path in images:
    img = image.load_img(img_path, target_size=(150, 150))
    x = image.img_to_array(img)
    x = x.reshape((1,) + x.shape)

    i = 0
    for batch in datagen.flow(x, batch_size=1, save_to_dir=output_image_folder, save_prefix='aug', save_format='jpg'):
        i += 1
        if i >= 4:  # сохранить 4 аугментированные версии каждого изображения
            break
