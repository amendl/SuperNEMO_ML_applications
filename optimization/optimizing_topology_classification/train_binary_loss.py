

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
task            = import_arbitrary_module("task","/sps/nemo/scratch/amendl/AI/my_lib/optimizing_topology_classification/scripts/dataset.py")


def push_attention_block(option_name,layer,use_normalization = False):
    ratio = int(os.environ[option_name])
    if ratio == 0:
        return layer
    else:
        if use_normalization:
            return keras.layers.concatenate([keras.layers.LayerNormalization(axis=[1,2,3])(attention_lib.cbam_block(layer,ratio)),layer],axis=-1)
        else:
            return keras.layers.concatenate([attention_lib.cbam_block(layer,ratio),layer],axis=-1)


def architecture_without_normalization():
    '''
    
    '''
    i = keras.Input(shape=(9,113))
    img = layers.Reshape((9,113,1))(i)
    x = layers.Conv2D(16,(3,15),activation = 'relu',padding="same")(img)
    x = layers.MaxPooling2D(pool_size = (1,2),strides=(1,2))(push_attention_block("P1",x))
    x = layers.Conv2D(128,(3,7),activation = 'relu',padding="same")(x)
    x = layers.MaxPooling2D(pool_size = (1,2),strides=(1,2))(push_attention_block("P2",x))
    x = layers.Conv2D(256,(3,3),activation = 'relu',padding="same")(x)
    x = layers.MaxPooling2D(pool_size = (2,2))(push_attention_block("P3",x))
    x = layers.Conv2D(256,(3,3),activation = 'relu',padding="same")(x) 
    x = layers.MaxPooling2D(pool_size = (2,2))(push_attention_block("P4",x))
    x = layers.Conv2D(256,(3,3),activation = 'relu',padding="same")(x) 
    x = layers.MaxPooling2D(pool_size = (2,2))(push_attention_block("P5",x)) 
    x = layers.Dropout(0.2)(x)
    x = layers.Flatten()(x)
    x = layers.Dense(256,activation='relu',use_bias=True)(x)
    x = layers.Dense(128,activation='relu',use_bias=True)(x)
    x = layers.Dense(4)(x)
    x = layers.Softmax()(x)

    model = keras.Model(inputs = i, outputs = x)

    return model

def architecture_with_normalization():
    '''
    
    '''
    i = keras.Input(shape=(9,113))
    img = layers.Reshape((9,113,1))(i)
    x = layers.Conv2D(256,(3,15),activation = 'relu',padding="same")(img)
    x = layers.MaxPooling2D(pool_size = (1,2),strides=(1,2))(push_attention_block("P1",x,True))
    x = layers.Conv2D(128,(3,7),activation = 'relu',padding="same")(x)
    x = layers.MaxPooling2D(pool_size = (1,2),strides=(1,2))(push_attention_block("P2",x,True))
    x = layers.Conv2D(128,(3,3),activation = 'relu',padding="same")(x)
    x = layers.MaxPooling2D(pool_size = (2,2))(push_attention_block("P3",x,True))
    x = layers.Conv2D(128,(3,3),activation = 'relu',padding="same")(x) 
    x = layers.MaxPooling2D(pool_size = (2,2))(push_attention_block("P4",x,True))
    x = layers.Conv2D(128,(3,3),activation = 'relu',padding="same")(x) 
    x = layers.MaxPooling2D(pool_size = (2,2))(push_attention_block("P5",x,True)) 
    x = layers.Dropout(0.2)(x)
    x = layers.Flatten()(x)
    x = layers.Dense(256,activation='relu',use_bias=True)(x)
    x = layers.Dense(128,activation='relu',use_bias=True)(x)
    x = layers.Dense(1,activation='sigmoid')(x)

    model = keras.Model(inputs = i, outputs = x)

    return model


if __name__=="__main__":



    tracks = 6
    files=10
    events = 4000

    model = architecture_with_normalization()

    print("Model summary:")
    model.summary()
    my_ml_lib.count_and_print_weights(model,True)


    keras.utils.plot_model(model = model,to_file = "model.png",show_shapes= True,show_dtype = True)

    
    dataset_size = tracks*files*events
    dataset = tf.data.Dataset.from_generator(
        generator = lambda: task.generator([1,2,3,4],[0,1,2,3,4,5,6,7],events),
        output_signature=(tf.TensorSpec(shape=(3),dtype=tf.int64))
    )
    dataset = dataset.shuffle(dataset_size,reshuffle_each_iteration = True)
    dataset = dataset.map(task.load_event)
    print(dataset)

    # for i in dataset:
    #     tf.print(i,summarize=-1)

    val_dataset = tf.data.Dataset.from_generator(
        generator = lambda: task.generator([1,2,3,4],[8],events),
        output_signature=(tf.TensorSpec(shape=(3),dtype=tf.int64))
    )
    val_dataset = val_dataset.map(task.load_event)

    model.compile(
        optimizer = 'adam',
        loss = tf.keras.losses.BinaryCrossentropy(from_logits=False),
        metrics = ['accuracy']
    )
    history = model.fit(
        x = dataset.batch(1024).prefetch(2),
        epochs = 10,
        validation_data = val_dataset.batch(1024).prefetch(2)
    )
    my_ml_lib.plot_train_val_accuracy(history)

    # test_dataset = tf.data.Dataset.from_generator(
    #     generator = lambda: task.generator([1,2,3,4],[9],events),
    #     output_signature=(tf.TensorSpec(shape=(3),dtype=tf.int64))
    # )
    # test_dataset = test_dataset.map(task.load_event)

    # my_ml_lib.confusion(model,test_dataset)

    model.save("model")

    with open("data_simple.txt","w") as f:
        f.write(str(history.history['val_accuracy'][-1]))

