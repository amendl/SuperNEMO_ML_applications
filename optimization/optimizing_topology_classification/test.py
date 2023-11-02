#! /sps/nemo/scratch/amendl/AI/virtual_env_python391/bin/python
import tensorrt
import tensorflow as tf
from tensorflow import keras
import numpy as np
import sys

import ROOT
import matplotlib.pyplot as plt


from keras import backend as K

import math

import sys




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

task  = import_arbitrary_module("task","/sps/nemo/scratch/amendl/AI/my_lib/optimizing_topology_classification/scripts/dataset.py")
attention_lib   = import_arbitrary_module("attention",'/sps/nemo/scratch/amendl/AI/my_lib/attention/cbam.py')
my_ml_lib       = import_arbitrary_module("my_ml_lib",'/sps/nemo/scratch/amendl/AI/my_lib/lib.py')



def push_attention_block(option_name,layer,use_normalization = False):
    ratio = int(os.environ[option_name])
    if ratio == 0:
        return layer
    else:
        if use_normalization:
            return keras.layers.concatenate([keras.layers.LayerNormalization(axis=[1,2,3])(attention_lib.cbam_block(layer,ratio)),layer],axis=-1)
        else:
            return keras.layers.concatenate([attention_lib.cbam_block(layer,ratio),layer],axis=-1)


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


def tf_to_numpy(tensor,matteo_shape=True):
    def non_vectorized_f(x):
        if x>0.5:
            return 1.
        else:
            return 0.
    vectorized_f = np.vectorize(non_vectorized_f)

    dataset = tf.data.Dataset.from_tensor_slices(tensor)
    iterator = dataset.as_numpy_iterator()
    elements = [element for element in iterator]
    x = np.vstack(elements)
    if matteo_shape:
        x = np.transpose(x)

        x = np.delete(x, (0,1,11), axis=0)
        x = np.delete(x,(0,1), axis=1)
    # x = vectorized_f(x)
    return x

def get_batched_dataset(i):
    return tf.data.Dataset.from_generator(
        generator = lambda: task.generator([i],[8],5000),
        output_signature=(tf.TensorSpec(shape=(3),dtype=tf.int64))
    ).map(task.load_event_testing).batch(100).prefetch(2)

def process_dataset(is_event,model,dataset):
    binary_accuracy = tf.keras.metrics.BinaryAccuracy()


    results = model.predict(dataset)
    if is_event:
        binary_accuracy.update_state(tf.ones_like(results),results)
    else:
        binary_accuracy.update_state(tf.zeros_like(results),results)
    return binary_accuracy.result().numpy()

if __name__ == "__main__":
    print("Loading model")
    sys.stdout.flush()
    model = architecture_with_normalization()
    model.load_weights("model/variables/variables")
    print("Initializing metrics")
    sys.stdout.flush()



    with open("data.txt",'w') as f:
        f.write("track_1:" + str(process_dataset(False,model,get_batched_dataset(1))))
        f.write("track_2:" + str(process_dataset(False,model,get_batched_dataset(2))))
        f.write("track_3:" + str(process_dataset(False,model,get_batched_dataset(3))))
        f.write("event_0:" + str(process_dataset(True,model,get_batched_dataset(4))))
        f.write("event_1:" + str(process_dataset(True,model,get_batched_dataset(5))))


        
    

    #     if i % 10 == 0:
    #         print(i,flush=True)
    #         sys.stdout.flush()
    # metrics.plot_less_than_expected("less_than_expected.pdf")
    # metrics.plot_more_than_ecpected("more_than_expected.pdf")


    #     if i % 10 == 0:
    #         print(i,flush=True)

    # print(0.005+0.01*np.argmax(finder.histo))
    # plt.plot(np.linspace(0.005,1.-0.005,num=100),finder.histo)
    # plt.axvline(0.005+0.01*float(np.argmax(finder.histo)),color="red",linestyle="dashed")
    # plt.xlabel("threshold")
    # plt.ylabel("score (the bigger the better)")
    # plt.savefig("find.pdf")    
