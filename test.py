import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model

# CIFAR-10 class names
class_names = ['Airplane', 'Automobile', 'Bird', 'Cat', 'Deer', 'Dog', 'Frog', 'Horse', 'Ship', 'Truck']


def load_and_predict(model_path, image_path):
    # Modeli yükle
    model = load_model(model_path)

    # Test görüntüsünü yükle ve işle
    img = tf.keras.preprocessing.image.load_img(image_path, target_size=(32, 32))
    img_array = tf.keras.preprocessing.image.img_to_array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)  # Modelin giriş boyutlarına uygun hale getir

    # Tahmin yap
    prediction = model.predict(img_array)
    predicted_class = np.argmax(prediction, axis=1)[0]
    predicted_probabilities = prediction[0]

    return predicted_class, predicted_probabilities


# Model ve test görüntüsü yolunu belirle
model_path = 'cifar10_model.h5'
test_image_path = '/Users/senakara/PycharmProjects/pythonProject62/dog4.png'  # Test görüntüsünün tam yolunu belirtin

# Tahmini yap ve sonucu yazdır
predicted_class_index, predicted_probabilities = load_and_predict(model_path, test_image_path)
predicted_class_name = class_names[predicted_class_index]

print(f'Predicted Class Index: {predicted_class_index}')
print(f'Predicted Class Name: {predicted_class_name}')

# Tüm sınıfların tahmin edilen olasılıklarını yazdır
for idx, (class_name, probability) in enumerate(zip(class_names, predicted_probabilities)):
    print(f'{class_name}: {probability:.4f}')
