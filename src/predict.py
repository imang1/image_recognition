import sys
import numpy as np
import tensorflow.contrib.keras as keras
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

path = sys.argv[1]

img = keras.preprocessing.image.load_img(path, grayscale=False, target_size=(299,299))
img = keras.preprocessing.image.img_to_array(img)
img = np.expand_dims(img, axis=0)
img = keras.applications.inception_v3.preprocess_input(img)

model = keras.models.load_model('new_product_photos_model.h5')

print('Model loaded')
# Print prediction

results = model.predict(img)

labels =  ['Apparel & Accessories',
           'Computers & Office',
           'Consumer Electronics',
           'General Merchandise',
           'Home & Garden',
           'Toys & Baby']
#labels  = keras.applications.inception_v3.decode_predictions(results, top=3)[0]
i = 0
for label in labels:
    print(label)
    for result in results:
        print('{0:.10f}'.format(result[i]))
        i += 1
    
