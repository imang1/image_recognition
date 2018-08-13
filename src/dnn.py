import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
from tensorflow.python.keras.preprocessing.image import ImageDataGenerator
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Conv2D, MaxPooling2D
from tensorflow.python.keras.layers import Activation, Dropout, Flatten, Dense
from tensorflow.python.keras import backend as K
from tensorflow.python.keras.callbacks import ModelCheckpoint, EarlyStopping, TensorBoard
import tensorflow.contrib.keras as keras

# dimensions of our images.
img_width, img_height = 299, 299

train_data_dir = 'downloads/train_wo_product'
nb_train_samples = 1203 
epochs = 20
batch_size = 16

input_shape = (img_width, img_height, 3)

PWD = '.'
MODEL = "{}/dnn_model.h5".format(PWD)

if os.path.isfile(MODEL):
    model = keras.models.load_model(MODEL)
else:
    model = Sequential()

    model.add(Flatten( input_shape=input_shape))
    model.add(Dense(16, activation='relu'))
    model.add(Dense(6, activation = 'sigmoid'))

    model.compile(loss='categorical_crossentropy',
                  optimizer='rmsprop',
                  metrics=['accuracy'])


train_datagen = ImageDataGenerator(
    rescale=1. / 255,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True)



train_generator = train_datagen.flow_from_directory(
    train_data_dir,
    target_size=(img_width, img_height),
    batch_size=batch_size,
    class_mode='categorical')


# save weights of best training epoch
callbacks_list = [
    ModelCheckpoint(MODEL, period=1),
    #EarlyStopping(monitor='val_acc', patience=5, verbose=0),
    TensorBoard(log_dir='tensorboard/inception-v3-train-top-layer', histogram_freq=0, write_graph=False, write_images=False)
    ]

model.fit_generator(
    train_generator,
    steps_per_epoch=nb_train_samples // batch_size ,
    epochs=epochs,
    callbacks=callbacks_list)
