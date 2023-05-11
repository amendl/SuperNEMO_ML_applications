

#TODO https://stats.stackexchange.com/questions/12702/what-are-the-measure-for-accuracy-of-multilabel-data

import datasets.number_of_tracks_my_generator as task

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


def plot_train_val_accuracy(history):
    plt.plot(history.history['accuracy'])
    plt.plot(history.history['val_accuracy'])
    plt.title('model accuracy')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(['train', 'val'], loc='upper left')
    plt.savefig(
        fname = 'training_accuracy.pdf',
        format = 'pdf'
    )

def confusion(model, test_dataset):
    y_true = []

    for _,label in test_dataset:
        y_true.append(tf.argmax(label))

    print("Creating confusion matrix")
    prediction=model.predict(test_dataset.batch(256))
    prediction = np.argmax(prediction, axis=1)
    cm = confusion_matrix(prediction, y_true)
    tf.print(cm,summarize=-1)
    # sn.heatmap(cm, annot=True, annot_kws={"size": 16})
    # plt.savefig("confusion_matrix.pdf")


def architectureTop():
    i = keras.Input(shape=(9,113))
    img = layers.Reshape((9,113,1))(i)
    x = layers.Conv2D(256,(3,15),activation = 'relu',padding="same")(img)
    x = layers.MaxPooling2D(pool_size = (1,2),strides=(1,2))(x)
    x = layers.Conv2D(256,(3,3),activation = 'relu',padding="same")(x)
    x = layers.MaxPooling2D(pool_size = (2,2))(x)
    x = layers.Conv2D(256,(3,3),activation = 'relu',padding="same")(x) 
    x = layers.MaxPooling2D(pool_size = (2,2))(x)
    x = layers.Conv2D(256,(3,3),activation = 'relu',padding="same")(x) 
    x = layers.MaxPooling2D(pool_size = (2,2))(x)
    flatten = layers.Flatten(name='flatten')(x)

    first_otuput = layers.Dense(128,activation='relu',use_bias=True,name='number_of_tracks_input')(flatten)
    first_otuput = layers.Dense(64,activation='sigmoid',use_bias=True)(first_otuput)
    first_otuput = layers.Dense(4)(first_otuput)
    first_otuput = layers.Softmax(name='number_of_tracks_output')(first_otuput)

    # row_part = layers.Dense(256,activation='relu',use_bias=True,name='row_part_input')(flatten)
    # row_part = layers.Dense(128,activation='relu',use_bias=True)(row_part)
    # row_part = layers.Dense( 64,activation='relu',use_bias=True)(row_part)
    # row_part = layers.Dense( 15,activation='sigmoid',use_bias=False,name='row_part_output')(row_part)

    column_part = layers.Dense(256,activation='relu',use_bias=True,name='column_part_input')(flatten)
    column_part = layers.Dense(128,activation='relu',use_bias=True)(column_part)
    column_part = layers.Dense( 128,activation='relu',use_bias=True)(column_part)
    column_part = layers.Dense( 22,activation='sigmoid',use_bias=False,name='column_part_output')(column_part)

    model = keras.Model(inputs = i, outputs = [first_otuput,column_part])

    return model

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
        raise Exception(f"[Custom exception {__file__}:strategy]: Not valid devices for tensorflow.distribute.MirroredStrategy nor valid device for tensorflow.distribute.OneDeviceStrategy were provided.")
    
def count_and_print_weights(model,_print=True):
    trainable_count = np.sum([K.count_params(w) for w in model.trainable_weights])
    non_trainable_count = np.sum([K.count_params(w) for w in model.non_trainable_weights])
    if _print:
        print(f'Total params: {trainable_count + non_trainable_count}')
        print(f'Trainable params: {trainable_count}')
        print(f'Non-trainable params: {non_trainable_count}')

    return trainable_count,non_trainable_count


if __name__=="__main__":
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
        
    '''
    
    '''
    tracks = 4
    files=8
    events = 5000
    dataset_size = tracks*files*events
    datetime_string = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")


    with strategy.scope():

        model = architectureTop()

        model.compile(
            optimizer = 'adam',
            loss = {
                'number_of_tracks_output'  : tf.keras.losses.CategoricalCrossentropy(),
#
                'column_part_output'            : tf.keras.losses.BinaryCrossentropy()
            },
            metrics = {
                'number_of_tracks_output'  : 'accuracy',
 #               'row'               : tf.keras.metrics.binary_crossentropy,
                'column_part_output'       : 'accuracy'
                      },
        )

        print("Model summary:")
        count_and_print_weights(model,True)

        dataset = tf.data.Dataset.from_generator(
            generator = lambda: task.generator(tracks,[0,1,2,3,4,5,6,7],events),
            output_signature=(tf.TensorSpec(shape=(3),dtype=tf.int64))
        )
        dataset = dataset.map(task.load_event).shuffle(dataset_size,reshuffle_each_iteration = True).batch(512).prefetch(1)

        val_dataset = tf.data.Dataset.from_generator(
            generator = lambda: task.generator(tracks,[8],events),
            output_signature=(tf.TensorSpec(shape=(3),dtype=tf.int64))
        )
        val_dataset = val_dataset.map(task.load_event).batch(512).prefetch(1)

        test_dataset = tf.data.Dataset.from_generator(
            generator = lambda: task.generator(tracks,[9],events),
            output_signature=(tf.TensorSpec(shape=(3),dtype=tf.int64))
        )
        test_dataset = test_dataset.map(task.load_event)

        log_dir = "fit_logs/" + datetime_string
        tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)

    history=None
    with tf.profiler.experimental.Profile("profiler/" + datetime_string):
        history = model.fit(
            x = dataset,
            epochs = 15,
            validation_data = val_dataset,
            callbacks=[tensorboard_callback]
        )
    plot_train_val_accuracy(history,"accuracy.pdf")

    model.save("model")
