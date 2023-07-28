import autoencoder_dataset as task

from sklearn.metrics import confusion_matrix

import numpy as np

import tensorflow as tf
from tensorflow import keras
from keras import layers # Dense, Droupout, Softmax, BatchNormalization, Conv2D
from keras.callbacks import EarlyStopping
import tensorflow.keras.backend as K

import matplotlib.pyplot as plt

import sys
import datetime

def count_and_print_weights(model,_print=True):
    trainable_count = np.sum([K.count_params(w) for w in model.trainable_weights])
    non_trainable_count = np.sum([K.count_params(w) for w in model.non_trainable_weights])
    if _print:
        print(f'Total params: {trainable_count + non_trainable_count}')
        print(f'Trainable params: {trainable_count}')
        print(f'Non-trainable params: {non_trainable_count}')

    return trainable_count,non_trainable_count


def process_command_line_arguments():
    '''
        returns:
            - device if --OneDeviceStategy otherwise None
            - devices if --MirroredStrategy otherwise 
    '''
    mode = 0
    device=None
    devices=[]
    for arg in sys.argv[1:]:
        if arg=="--OneDeviceStrategy":
            mode=1
        elif arg=="--MirroredStrategy":
            mode=2
        elif mode==1:
            device=arg
        elif mode==2:
            devices.append(arg)
        else:
            raise Exception(f"[Custom exception {__file__}:process_command_line_arguments]: \"{arg}\" is not valid parameter for this script.")
        
    return (device,devices)


def choose_strategy(device,devices=None):
    if devices is not None and devices:
        return tf.distribute.MirroredStrategy(devices)
    elif device is not None:
        return tf.distribute.OneDeviceStrategy(device)
    else:
        raise Exception(f"[Custom exception {__file__}:choose_strategy]: No valid devices for tensorflow.distribute.MirroredStrategy nor valid device for tensorflow.distribute.OneDeviceStrategy were provided.")

def matteo_with_skip_connection():
    '''
        Original model by matteo with skip connections
    '''
    
    img_input = keras.layers.Input(shape=(116, 12))
    a = keras.layers.Reshape((116, 12,1))(img_input)
    conv1 = keras.layers.Conv2D(32, (3, 3), activation='relu', padding='same')(a)
    conv1 = keras.layers.Dropout(0.2)(conv1)
    conv1 = keras.layers.Conv2D(32, (3, 3), activation='relu', padding='same')(conv1)
    pool1 = keras.layers.MaxPooling2D((2, 2))(conv1)

    conv2 = keras.layers.Conv2D(64, (3, 3), activation='relu', padding='same')(pool1)
    conv2 = keras.layers.Dropout(0.2)(conv2)
    conv2 = keras.layers.Conv2D(64, (3, 3), activation='relu', padding='same')(conv2)
    pool2 = keras.layers.MaxPooling2D((2, 2), padding='same')(conv2)

    conv3 = keras.layers.Conv2D(128, (3, 3), activation='relu', padding='same')(pool2)
    conv3 = keras.layers.Dropout(0.2)(conv3)
    conv3 = keras.layers.Conv2D(128, (3, 3), activation='relu', padding='same')(conv3)

    up1 = keras.layers.concatenate([keras.layers.UpSampling2D((2, 2))(conv3), conv2], axis=-1)
    conv4 = keras.layers.Conv2D(64, (3, 3), activation='relu', padding='same')(up1)
    conv4 = keras.layers.Dropout(0.2)(conv4)
    conv4 = keras.layers.Conv2D(64, (3, 3), activation='relu', padding='same')(conv4)

    up2 = keras.layers.concatenate([keras.layers.UpSampling2D((2, 2))(conv4), conv1], axis=-1)
    conv5 = keras.layers.Conv2D(32, (3, 3), activation='relu', padding='same')(up2)
    conv5 = keras.layers.Dropout(0.2)(conv5)
    conv5 = keras.layers.Conv2D(32, (3, 3), activation='relu', padding='same')(conv5)

    out = keras.layers.Conv2D(1, 3, padding='same')(keras.layers.UpSampling2D(1)(conv5)) # (2,3.)

    model = keras.models.Model(img_input, out)

    return model


def matteo_without_skip_connection():
    '''
        Model by matteo wit hskip connections removed. To use matteos original model 'see matteo_with_skip_connection'
    '''
    img_input = keras.layers.Input(shape=(116, 12))
    a = keras.layers.Reshape((116, 12,1))(img_input)

    conv1 = keras.layers.Conv2D(32, (3, 3), activation='relu', padding='same')(a)
    conv1 = keras.layers.Dropout(0.2)(conv1)
    conv1 = keras.layers.Conv2D(32, (3, 3), activation='relu', padding='same')(conv1)
    pool1 = keras.layers.MaxPooling2D((2, 2))(conv1)

    conv2 = keras.layers.Conv2D(64, (3, 3), activation='relu', padding='same')(pool1)
    conv2 = keras.layers.Dropout(0.2)(conv2)
    conv2 = keras.layers.Conv2D(64, (3, 3), activation='relu', padding='same')(conv2)
    pool2 = keras.layers.MaxPooling2D((2, 2), padding='same')(conv2)

    conv3 = keras.layers.Conv2D(128, (3, 3), activation='relu', padding='same')(pool2)
    conv3 = keras.layers.Dropout(0.2)(conv3)
    conv3 = keras.layers.Conv2D(128, (3, 3), activation='relu', padding='same')(conv3)

    up1 = keras.layers.UpSampling2D((2, 2))(conv3)
    conv4 = keras.layers.Conv2D(64, (3, 3), activation='relu', padding='same')(up1)
    conv4 = keras.layers.Dropout(0.2)(conv4)
    conv4 = keras.layers.Conv2D(64, (3, 3), activation='relu', padding='same')(conv4)

    up2 = keras.layers.UpSampling2D((2, 2))(conv4)
    conv5 = keras.layers.Conv2D(32, (3, 3), activation='relu', padding='same')(up2)
    conv5 = keras.layers.Dropout(0.2)(conv5)
    conv5 = keras.layers.Conv2D(32, (3, 3), activation='relu', padding='same')(conv5)

    out = keras.layers.Conv2D(1, 3, padding='same')(keras.layers.UpSampling2D(1)(conv5)) # (2,3)

    model = keras.models.Model(img_input, out)

    return model

if __name__=="__main__":
    tracks = 1
    files = 10
    events = 10000
    dataset_size = tracks*files*events
    datetime_string = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")

    print(tf.config.list_physical_devices())

    device,devices = process_command_line_arguments()
    strategy = choose_strategy(device,devices)
    # print(f"Strategy arguments {device}; {devices}")
    print("Running with strategy: ",str(strategy))
    try:
        print(" on device ", strategy.device)
    except:
        pass
    try:
        print(" on devices ",strategy.devices)
    except:
        pass

    tf.keras.utils.disable_interactive_logging()

    with strategy.scope():
        model = matteo_without_skip_connection()

        model.compile(
            optimizer='adam',
            loss=tf.keras.losses.BinaryCrossentropy(),
            metrics='accuracy'
        )

        print("Model summary:")
        model.summary()
        count_and_print_weights(model,True)

        dataset = tf.data.Dataset.from_generator(
            generator = lambda: task.generator([1,2,3,4],[0,1,2,3,4,5,6,7],events),
            output_signature=(tf.TensorSpec(shape=(3),dtype=tf.int32))
        )
        dataset = dataset.map(task.load_event).shuffle(dataset_size,reshuffle_each_iteration = True)

        val_dataset = tf.data.Dataset.from_generator(
            generator = lambda: task.generator([1,2,3,4],[8],events),
            output_signature=(tf.TensorSpec(shape=(3),dtype=tf.int32))
        )
        val_dataset = val_dataset.map(task.load_event)

        test_dataset = tf.data.Dataset.from_generator(
            generator = lambda: task.generator([1,2,3,4],[9],events),
            output_signature=(tf.TensorSpec(shape=(3),dtype=tf.int32))
        )
        test_dataset = test_dataset.map(task.load_event)


    history = model.fit(
        x = dataset.batch(128).prefetch(1),
        epochs = 8,
        validation_data = val_dataset.batch(128).prefetch(1)
    )

    model.save("model")





