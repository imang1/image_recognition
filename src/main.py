import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
from tensorflow.python.keras.layers import Dense, GlobalMaxPooling2D, Dropout
from tensorflow.python.keras.applications import InceptionV3
from tensorflow.python.keras.models import Model
from tensorflow.python.keras.preprocessing.image import ImageDataGenerator
from tensorflow.python.keras.callbacks import ModelCheckpoint, EarlyStopping, TensorBoard
from tensorflow.python.keras import backend as K


# hyper parameters for model
nb_classes = 6  # number of classes
based_model_last_block_layer_number = 126  # value is based on base model of Inception V3
img_width, img_height = 299, 299  
batch_size = 4 #weak CPU memory capacity, can't use larger than 4
nb_epoch = 50  # number of iteration the algorithm gets trained.
transformation_ratio = .05  # how aggressive will be the data augmentation/transformation
nb_train_samples = 6286# Total number of train samples
nb_validation_samples = 1570  # Total number of test samples.

PWD = 'src'
MODEL = "{}/model.h5".format(PWD)

train_data_dir = 'downloads/train'
validation_data_dir =  'downloads/test'
model_path = '.'

    
if K.image_data_format() == 'channels_first':
    input_shape = (3, img_width, img_height)
else:
    input_shape = (img_width, img_height, 3)


if os.path.isfile(MODEL):
    model = load_model(MODEL)
else:
    base_model = InceptionV3(input_shape=input_shape, weights='imagenet', include_top=False)
    
    x = base_model.output
    x = GlobalMaxPooling2D()(x)
    x = Dense(128, activation='relu', name='fc1')(x)
    x = Dropout(0.2)(x)

    predictions = Dense(nb_classes, activation='softmax', name='predictions')(x)

    # add top layer block to your base model
    model = Model(base_model.input, predictions)
    # freeze all layers of the based model that is already pre-trained.
    for layer in base_model.layers:
        layer.trainable = False
                    
        model.compile(optimizer='nadam',
                loss='categorical_crossentropy', 
                metrics=['accuracy'])
        


train_datagen = ImageDataGenerator(rescale=1. / 255,
                                   rotation_range=transformation_ratio,
                                   shear_range=transformation_ratio,
                                   zoom_range=transformation_ratio,
                                   cval=transformation_ratio,
                                   horizontal_flip=True,
                                   vertical_flip=True)

validation_datagen = ImageDataGenerator(rescale=1. / 255)


train_generator = train_datagen.flow_from_directory(train_data_dir,
                                                    target_size=[img_width, img_height],
                                                    batch_size=batch_size,
                                                    class_mode='categorical')
  

validation_generator = validation_datagen.flow_from_directory(validation_data_dir,
                                                              target_size=[img_width, img_height],
                                                              batch_size=batch_size,
                                                              class_mode='categorical')



# save weights of best training epoch
callbacks_list = [
    ModelCheckpoint(MODEL, monitor='val_acc', verbose=1, save_best_only=True),
    EarlyStopping(monitor='val_acc', patience=5, verbose=0),
    TensorBoard(log_dir='tensorboard/inception-v3-train-top-layer', histogram_freq=0, write_graph=False, write_images=False)
    ]

# Train Simple CNN
model.fit_generator(train_generator,
                    steps_per_epoch=nb_train_samples // batch_size,
                    epochs=nb_epoch / 5,
                    validation_data=validation_generator,
                    validation_steps=nb_validation_samples // batch_size,
                    callbacks=callbacks_list)


