import clustering_dataset as task

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


def import_arbitraty_module(module_name,path):
    import importlib.util
    import sys

    spec = importlib.util.spec_from_file_location(module_name,path)
    imported_module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = imported_module
    spec.loader.exec_module(imported_module)

    return imported_module


matteo    = import_arbitraty_module("matteo","/sps/nemo/scratch/amendl/AI/my_lib/latent_space_tricks/matteo.py")
my_ml_lib = import_arbitraty_module("my_ml_lib",'/sps/nemo/scratch/amendl/AI/my_lib/lib.py')


if __name__=="__main__":
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
        model = matteo.without_skip_connection()

        model.compile(
            optimizer='adam',
            loss=tf.keras.losses.BinaryCrossentropy(),
            metrics='accuracy'
        )

        print("Model summary:")
        model.summary()
        my_ml_lib.count_and_print_weights(model,True)


        dataset = tf.data.Dataset.from_generator(
            generator = lambda: task.generator([2],[0,1,2,3,4,5,6,7],events),
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

