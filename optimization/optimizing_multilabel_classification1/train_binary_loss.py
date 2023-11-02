

import tensorrt
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
import os 



def import_arbitrary_module(module_name,path):
    import importlib.util
    import sys

    spec = importlib.util.spec_from_file_location(module_name,path)
    imported_module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = imported_module
    spec.loader.exec_module(imported_module)

    return imported_module

attention_lib   = import_arbitrary_module("attention",'/sps/nemo/scratch/amendl/AI/my_lib/attention/cbam.py')
my_ml_lib       = import_arbitrary_module("my_ml_lib",'/sps/nemo/scratch/amendl/AI/my_lib/lib.py')
task            = import_arbitrary_module("task","/sps/nemo/scratch/amendl/AI/my_lib/optimizing_multilabel_classification1/scripts/dataset.py")





def push_attention_block(option_name,layer,use_normalization = False):
    ratio = int(os.environ[option_name])
    if ratio == 0:
        return layer
    else:
        if use_normalization:
            return keras.layers.concatenate([keras.layers.LayerNormalization(axis=[1,2,3])(attention_lib.cbam_block(layer,ratio)),layer],axis=-1)
        else:
            return keras.layers.concatenate([attention_lib.cbam_block(layer,ratio),layer],axis=-1)



def my_loss(i,axis=3):

    def my_cathegorical_crossentropy(target,output):
        output = output / tf.reduce_sum(output, axis, True)
        epsilon_ = tf.constant(K.epsilon(), output.dtype.base_dtype)
        output = tf.clip_by_value(output, epsilon_, 1.0 - epsilon_)
        return -tf.reduce_sum(tf.reduce_sum(target*tf.math.log(output),axis=3)*tf.constant(i))

    return my_cathegorical_crossentropy


def custom_keras_loss_lambda(arguments):
    input,y_true,y_predicted = arguments
    a = K.cumsum(y_true*tf.math.log(y_predicted),axis=3)
    return -K.cumsum(K.cumsum(K.cumsum(a,axis=0),axis=1),axis=2)



def my_cathegorical_crossentropy(target,output):
    output = output / tf.reduce_sum(output, 3, True)
    epsilon_ = tf.constant(K.epsilon(), output.dtype.base_dtype)
    output = tf.clip_by_value(output, epsilon_, 1.0 - epsilon_)
    return -tf.reduce_sum(target*tf.math.log(output),[1,2,3])



def cbam_optimization_layer_normalization(max_number_of_tracks,compile=False):

    img_input = keras.layers.Input(shape=(116, 12))
    img_true = keras.layers.Input(shape=(116, 12,max_number_of_tracks))
    a = keras.layers.Reshape((116, 12,1))(img_input)
    conv1 = keras.layers.Conv2D(32, (3, 3), activation='relu', padding='same')(a)
    conv1 = keras.layers.Dropout(0.2)(conv1)
    conv1 = keras.layers.Conv2D(32, (3, 3), activation='relu', padding='same')(conv1)
    pool1 = keras.layers.MaxPooling2D((2, 2))(push_attention_block("P1",conv1,True))

    conv2 = keras.layers.Conv2D(64, (3, 3), activation='relu', padding='same')(pool1)
    conv2 = keras.layers.Dropout(0.2)(conv2)
    conv2 = keras.layers.Conv2D(64, (3, 3), activation='relu', padding='same')(conv2)
    pool2 = keras.layers.MaxPooling2D((2, 2), padding='same')(push_attention_block("P2",conv2,True))

    conv3 = keras.layers.Conv2D(128, (3, 3), activation='relu', padding='same')(pool2)
    conv3 = keras.layers.Dropout(0.2)(conv3)
    conv3 = keras.layers.Conv2D(128, (3, 3), activation='relu', padding='same')(conv3)


    up1 = keras.layers.concatenate([keras.layers.UpSampling2D((2, 2))(push_attention_block("P3",conv3,True)), conv2], axis=-1)
    conv4 = keras.layers.Conv2D(64, (3, 3), activation='relu', padding='same')(up1)
    conv4 = keras.layers.Dropout(0.2)(push_attention_block("P4",conv4,True))
    conv4 = keras.layers.Conv2D(64, (3, 3), activation='relu', padding='same')(conv4)

    up2 = keras.layers.concatenate([keras.layers.UpSampling2D((2, 2))(conv4), conv1], axis=-1)
    conv5 = keras.layers.Conv2D(32, (3, 3), activation='relu', padding='same')(up2)
    conv5 = keras.layers.Dropout(0.2)(push_attention_block("P5",conv5,True))
    conv5 = keras.layers.Conv2D(32, (3, 3), activation='relu', padding='same')(conv5)

    out = keras.layers.Conv2D(max_number_of_tracks, 3, padding='same',activation='sigmoid')(keras.layers.UpSampling2D(1)(conv5))
    out = keras.layers.Softmax()(out)

    model = keras.models.Model(img_input, out)

    return model



def cbam_optimization_layer_normalization_big(max_number_of_tracks,compile=False):

    img_input = keras.layers.Input(shape=(116, 12))
    img_true = keras.layers.Input(shape=(116, 12,max_number_of_tracks))
    a = keras.layers.Reshape((116, 12,1))(img_input)
    conv1 = keras.layers.Conv2D(32*2, (3, 3), activation='relu', padding='same')(a)
    conv1 = keras.layers.Dropout(0.2)(conv1)
    conv1 = keras.layers.Conv2D(32*2, (3, 3), activation='relu', padding='same')(conv1)
    pool1 = keras.layers.MaxPooling2D((2, 2))(push_attention_block("P1",conv1,True))

    conv2 = keras.layers.Conv2D(64*2, (3, 3), activation='relu', padding='same')(pool1)
    conv2 = keras.layers.Dropout(0.2)(conv2)
    conv2 = keras.layers.Conv2D(64*2, (3, 3), activation='relu', padding='same')(conv2)
    pool2 = keras.layers.MaxPooling2D((2, 2), padding='same')(push_attention_block("P2",conv2,True))

    conv3 = keras.layers.Conv2D(128*2, (3, 3), activation='relu', padding='same')(pool2)
    conv3 = keras.layers.Dropout(0.2)(conv3)
    conv3 = keras.layers.Conv2D(128*2, (3, 3), activation='relu', padding='same')(conv3)


    up1 = keras.layers.concatenate([keras.layers.UpSampling2D((2, 2))(push_attention_block("P3",conv3,True)), conv2], axis=-1)
    conv4 = keras.layers.Conv2D(64*2, (3, 3), activation='relu', padding='same')(up1)
    conv4 = keras.layers.Dropout(0.2)(push_attention_block("P4",conv4,True))
    conv4 = keras.layers.Conv2D(64*2, (3, 3), activation='relu', padding='same')(conv4)

    up2 = keras.layers.concatenate([keras.layers.UpSampling2D((2, 2))(conv4), conv1], axis=-1)
    conv5 = keras.layers.Conv2D(32*2, (3, 3), activation='relu', padding='same')(up2)
    conv5 = keras.layers.Dropout(0.2)(push_attention_block("P5",conv5,True))
    conv5 = keras.layers.Conv2D(32*2, (3, 3), activation='relu', padding='same')(conv5)

    out = keras.layers.Conv2D(max_number_of_tracks, 3, padding='same',activation='sigmoid')(keras.layers.UpSampling2D(1)(conv5))
    out = keras.layers.Softmax()(out)

    model = keras.models.Model(img_input, out)

    return model



if __name__=="__main__":

    print("************************************")
    print("************************************")
    print(f"***     MAX_NUMBER_OF_TRACKS={task.MAX_NUMBER_OF_TRACKS}")
    print("************************************")
    print("************************************")

    tracks = 1
    files = 10
    events = 10000
    dataset_size = tracks*files*events
    datetime_string = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")

    print(tf.config.list_physical_devices())

    device,devices = my_ml_lib.process_command_line_arguments()
    strategy = my_ml_lib.choose_strategy(device,devices)
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

    with strategy.scope():

        model = cbam_optimization_layer_normalization_big(task.MAX_NUMBER_OF_TRACKS,compile=True)

        model.compile(
            optimizer='adam',
            loss=my_cathegorical_crossentropy,
            metrics=tf.keras.metrics.BinaryAccuracy()
        )

        print("Model summary:")
        model.summary()
        my_ml_lib.count_and_print_weights(model,True)

        dataset = tf.data.Dataset.from_generator(
            generator = lambda: task.generator([task.MAX_NUMBER_OF_TRACKS],[0,1,2,3,4,5,6,7],events), # range(1,task.MAX_NUMBER_OF_TRACKS+1)
            output_signature=(tf.TensorSpec(shape=(3),dtype=tf.int32))
        )
        dataset = dataset.shuffle(dataset_size,reshuffle_each_iteration = True).map(task.load_event)

        val_dataset = tf.data.Dataset.from_generator(
            generator = lambda: task.generator([task.MAX_NUMBER_OF_TRACKS],[8],events), # range(1,task.MAX_NUMBER_OF_TRACKS+1)
            output_signature=(tf.TensorSpec(shape=(3),dtype=tf.int32))
        )
        val_dataset = val_dataset.map(task.load_event)


    history = model.fit(
        x = dataset.batch(128).prefetch(2),
        epochs = 10,
        validation_data = val_dataset.batch(128).prefetch(2)
    )

    model.save("model")

