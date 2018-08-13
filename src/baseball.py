import numpy as np
import tensorflow.contrib.keras as keras
from sklearn.metrics import classification_report, confusion_matrix
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

PWD       = '.'
MODEL     = "{}/new_product_photos_model.h5".format(PWD)
THRESHOLD = 0.5

# Load model.
model = keras.models.load_model(MODEL)

# Stream test photos.
test_datagen = keras.preprocessing.image.ImageDataGenerator(
    rescale=1. / 255)

test_generator = test_datagen.flow_from_directory(
    "{}/downloads/test".format(PWD),
    target_size=(299, 299),
    batch_size=32,
    class_mode='categorical',
    shuffle=False)

# Test model.
results = model.predict_generator(test_generator)

root_names = ['Apparel & Accessories',
              'Computers & Office',
              'Consumer Electronics',
              'General Merchandise',
              'Home & Garden',
              'Toys & Baby']
count = 0.0
total = 0.0

predicted_list = list()
actual_list = list()

for i in range(0, len(results)):
    predicted = np.argmax(results[i])
    value     = results[i][predicted]
    actual    = test_generator.classes[i]

    predicted_list.append(predicted)
    actual_list.append(actual)

    if value > THRESHOLD:
        if predicted == actual:
            count = count + 1
        total = total + 1

    
print("accuracy: {}".format(count / total))
print("coverage: {}".format(total / len(results)))
print(classification_report(actual_list, predicted_list, target_names=root_names))
#print("Actual: {}".format(actual_list))
#print("Predictions: {}".format(predicted_list))
print(confusion_matrix(actual_list, predicted_list, labels = [0,1,2,3,4,5] ))
