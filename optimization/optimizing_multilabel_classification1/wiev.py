
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




def cbam_optimization_layer_normalization(max_number_of_tracks):

    img_input = keras.layers.Input(shape=(116, 12))
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

def my_print(tensor):
    # print(tensor.shape)
    for i in range(116):
        print(chr(9608),end="")
    print()
    for i in range(tensor.shape[0]):
        print(chr(9608),end="")
        for j in range(tensor.shape[1]):
            if tensor[i,j]>0.545:
                print(chr(0x2299),end="")
            else:
                print(" ",end="")
        print(chr(9608))
    for i in range(116):
        print(chr(9608),end="")



def analyse_event(ID,model,fillers):
    original,truth = task.load_event_helper(ID)
    model_output = model(tf.reshape(original,[1,116,12]))
    # if fillers[0].fill(tf_to_numpy(tf.reshape(original,[116,12])),tf_to_numpy(tf.reshape(truth,[116,12])),tf_to_numpy(tf.reshape(model_output,[116,12]))) > 25:
    #     draw_event(ID,model)
    #     input("Failed event found, continue?")
    # for f in fillers:
    #     f.fill(tf_to_numpy(tf.reshape(original,[116,12])),tf_to_numpy(tf.reshape(truth,[116,12])),tf_to_numpy(tf.reshape(model_output,[116,12])))
    
    my_print(tf_to_numpy(tf.reshape(original,[116,12])))
    print()
    # tf.print(tf.constant(tf_to_numpy(tf.reshape(tf.math.argmax(model_output,axis=3),[116,12]))),summarize=-1)
    tf.print(tf.constant(tf_to_numpy(tf.reshape(model_output.numpy()[:,:,:,4],[116,12]))),summarize=-1)
    input("")

if __name__=="__main__":


    model = cbam_optimization_layer_normalization(4)
    model.load_weights("model/variables/variables")


    name = "final"
    print(model)
    for i in range(5000): 
        id = [3,9,i]
        # if i % 100 == 0:
        print(id)
        sys.stdout.flush()
        analyse_event(tf.constant(id),model,None)











