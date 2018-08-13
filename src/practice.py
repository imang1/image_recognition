import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
#import sys
from tensorflow.python.keras.layers import Dense, GlobalMaxPooling2D, Dropout
#from tensorflow.python.keras.optimizers import *
from tensorflow.python.keras.applications import InceptionV3
from tensorflow.python.keras.models import Model
from tensorflow.python.keras.preprocessing.image import ImageDataGenerator
from tensorflow.python.keras.callbacks import ModelCheckpoint, EarlyStopping, TensorBoard
from tensorflow.python.keras import backend as K
#from tensorflow.python.keras 

# fix seed for reproducible results (only works on CPU, not GPU)
#seed = 9
#np.random.seed(seed=seed)
#tf.set_random_seed(seed=seed)

# hyper parameters for model
nb_classes = 6  # number of classes
based_model_last_block_layer_number = 126  # value is based on based model selected.
img_width, img_height = 299, 299  # change based on the shape/structure of your images
batch_size = 4  # try 4, 8, 16, 32, 64, 128, 256 dependent on CPU/GPU memory capacity (powers of 2 values).
nb_epoch = 50  # number of iteration the algorithm gets trained.
#learn_rate = 1e-4  # sgd learning rate
#momentum = .9  # sgd momentum to avoid local minimum
transformation_ratio = .05  # how aggressive will be the data augmentation/transformation
nb_train_samples = 6286# Total number of train samples. NOT including augmented images
nb_validation_samples = 1570  # Total number of train samples. NOT including augmented images.

PWD = 'src'
MODEL = "{}/model.h5".format(PWD)


def train(train_data_dir, validation_data_dir, model_path):
    # Pre-Trained CNN Model using imagenet dataset for pre-trained weights
    # base_model = Xception(input_shape=(img_width, img_height, 3), weights='imagenet', include_top=False)
    
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

            # add your top layer block to your base model
            model = Model(base_model.input, predictions)
            #print(model.summary())


            # first: train only the top layers (which were randomly initialized)
            # i.e. freeze all layers of the based model that is already pre-trained.
            for layer in base_model.layers:
                    layer.trainable = False
                    
            model.compile(optimizer='nadam',
                  loss='categorical_crossentropy',  # categorical_crossentropy if multi-class classifier
                  metrics=['accuracy'])
        

    # Read Data and Augment it: Make sure to select augmentations that are appropriate to your images.
    # To save augmentations un-comment save lines and add to your flow parameters.
    train_datagen = ImageDataGenerator(rescale=1. / 255,
                                       rotation_range=transformation_ratio,
                                       shear_range=transformation_ratio,
                                       zoom_range=transformation_ratio,
                                       cval=transformation_ratio,
                                       horizontal_flip=True,
                                       vertical_flip=True)

    validation_datagen = ImageDataGenerator(rescale=1. / 255)

    # os.makedirs(os.path.join(os.path.abspath(train_data_dir), 'preview'), exist_ok=True)
    train_generator = train_datagen.flow_from_directory(train_data_dir,
                                                        target_size=[img_width, img_height],
                                                        batch_size=batch_size,
                                                        class_mode='categorical')
    # save_to_dir=os.path.join(os.path.abspath(train_data_dir), '../preview')
    # save_prefix='aug',
    # save_format='jpeg')
    # use the above 3 commented lines if you want to save and look at how the data augmentations look like

    validation_generator = validation_datagen.flow_from_directory(validation_data_dir,
                                                                  target_size=[img_width, img_height],
                                                                  batch_size=batch_size,
                                                                  class_mode='categorical')



    # save weights of best training epoch: monitor either val_loss or val_acc

    #top_weights_path = os.path.join(os.path.abspath(model_path), 'top_model_weights.h5')
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

    # verbose
    #print("\nStarting to Fine Tune Model\n")

    # add the best weights from the train top model
    # at this point we have the pre-train weights of the base model and the trained weight of the new/added top model
    # we re-load model weights to ensure the best epoch is selected and not the last one.
    #model.load_weights(top_weights_path)

    # based_model_last_block_layer_number points to the layer in your model you want to train.
    # For example if you want to train the last block of a 19 layer VGG16 model this should be 15
    # If you want to train the last Two blocks of an Inception model it should be 172
    # layers before this number will used the pre-trained weights, layers above and including this number
    # will be re-trained based on the new data.
    #for layer in model.layers[:based_model_last_block_layer_number]:
    #    layer.trainable = False
    #for layer in model.layers[based_model_last_block_layer_number:]:
    #    layer.trainable = True

    # compile the model with a SGD/momentum optimizer
    # and a very slow learning rate.
    #model.compile(optimizer='nadam',
    #              loss='categorical_crossentropy',
    #              metrics=['accuracy'])
                  
    
    # save weights of best training epoch: monitor either val_loss or val_acc
    #final_weights_path = os.path.join(os.path.abspath(model_path), 'model_weights.h5')
    #callbacks_list = [
    #    ModelCheckpoint(final_weights_path, monitor='val_acc', verbose=1, save_best_only=True),
    #    EarlyStopping(monitor='val_loss', patience=5, verbose=0),
        #keras.callbacks.TensorBoard(log_dir='tensorboard/inception-v3-fine-tune', histogram_freq=0, write_graph=False, write_images=False)
    #]
    



    # fine-tune the model
    #model.fit_generator(train_generator,
    #                    steps_per_epoch=nb_train_samples // batch_size,
    #                    epochs=nb_epoch,
    #                    validation_data=validation_generator,
    #                    validation_steps=nb_validation_samples // batch_size)

    # save model
    #model_json = model.to_json()
    #with open(os.path.join(os.path.abspath(model_path), 'model.json'), 'w') as json_file:
    #    json_file.write(model_json)


train('downloads/train', 'downloads/test', '.')
