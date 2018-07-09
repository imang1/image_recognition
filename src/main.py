import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
from tensorflow.python.keras.layers import Dense, GlobalMaxPooling2D, Dropout
from tensorflow.python.keras.applications import InceptionV3
from tensorflow.python.keras.models import Model
from tensorflow.python.keras.preprocessing.image import ImageDataGenerator
from tensorflow.python.keras.callbacks import ModelCheckpoint, EarlyStopping, TensorBoard
from tensorflow.python.keras import backend as K
import tensorflow.contrib.keras as keras


# hyper parameters for model
nb_classes = 6  # number of classes
based_model_last_block_layer_number = 126  # value is based on base model of Inception V3
img_width, img_height = 299, 299  
batch_size = 16 #weak CPU memory capacity
nb_epoch = 5  # number of iteration the algorithm gets trained.
transformation_ratio = .05  # how aggressive will be the data augmentation/transformation
nb_train_samples = 1207 # Total number of train samples


PWD = '.'
MODEL = "{}/model.h5".format(PWD)

train_data_dir = 'downloads/train'
model_path = '.'

    
if K.image_data_format() == 'channels_first':
    input_shape = (3, img_width, img_height)
else:
    input_shape = (img_width, img_height, 3)


if os.path.isfile(MODEL):
    model = keras.models.load_model(MODEL)
else:
    base_model = InceptionV3(input_shape=input_shape, weights='imagenet', include_top=False, pooling='avg')
    
    # freeze all layers of the based model that is already pre-trained.
    for layer in base_model.layers:
        layer.trainable = False

    x = base_model.output
    #x = GlobalMaxPooling2D()(x)
    x = Dense(128, activation='relu')(x)
    x = Dropout(0.2)(x)

    predictions = Dense(nb_classes, activation='softmax', name='predictions')(x)

    # add top layer block to your base model
    model = Model(inputs=base_model.input, outputs=predictions)
                    
    model.compile(optimizer='adam',
                  loss='categorical_crossentropy', 
                  metrics=['accuracy'])
        


train_datagen = ImageDataGenerator(rescale=1. / 255,
                                   rotation_range=transformation_ratio,
                                   shear_range=transformation_ratio,
                                   zoom_range=transformation_ratio,
                                   cval=transformation_ratio,
                                   width_shift_range=transformation_ratio,
                                   height_shift_range=transformation_ratio,
                                   horizontal_flip=True,
                                   vertical_flip=True)



train_generator = train_datagen.flow_from_directory(train_data_dir,
                                                    target_size=[img_width, img_height],
                                                    batch_size=batch_size,
                                                    class_mode='categorical')
                                                        



# save weights of best training epoch
callbacks_list = [
    ModelCheckpoint(MODEL, period=1),
    TensorBoard(log_dir='tensorboard/inception-v3-train-top-layer', histogram_freq=0, write_graph=False, write_images=False)
    ]

# Train Simple CNN
model.fit_generator(train_generator,
                    steps_per_epoch=nb_train_samples // batch_size,
                    epochs=nb_epoch,
                    callbacks=callbacks_list)


