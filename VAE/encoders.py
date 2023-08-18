


import tensorflow as tf
from tensorflow import keras
from keras import layers


def import_arbitrary_module(module_name,path):
    import importlib.util
    import sys

    spec = importlib.util.spec_from_file_location(module_name,path)
    imported_module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = imported_module
    spec.loader.exec_module(imported_module)

    return imported_module

vaes            = import_arbitrary_module("vaes",           "/sps/nemo/scratch/amendl/AI/my_lib/latent_space_tricks/VAE/lib.py")
my_ml_lib       = import_arbitrary_module("my_ml_lib",      "/sps/nemo/scratch/amendl/AI/my_lib/lib.py")


def architecture1(latent_dim):
    '''
        architecture based on encoders used by me for number of tracks classification problem and decoder taken from matteos autoencoder architecture
    '''
    i = keras.Input(shape=(9,113))
    img = layers.Reshape((9,113,1))(i)
    x = layers.Conv2D(256,(3,15),activation = 'relu',padding="same")(img)
    x = layers.MaxPooling2D(pool_size = (1,2),strides=(1,2))(x)
    x = layers.Conv2D(256,(3,7),activation = 'relu',padding="same")(x)
    x = layers.MaxPooling2D(pool_size = (1,2),strides=(1,2))(x)
    x = layers.Conv2D(256,(3,3),activation = 'relu',padding="same")(x)
    x = layers.MaxPooling2D(pool_size = (2,2))(x)
    x = layers.Conv2D(256,(3,3),activation = 'relu',padding="same")(x) 
    # x = layers.MaxPooling2D(pool_size = (2,2))(x)
    # x = layers.Conv2D(256,(3,3),activation = 'relu',padding="same")(x) 
    # x = layers.MaxPooling2D(pool_size = (2,2))(x) 
    x = layers.Dropout(0.2)(x)
    x = layers.Flatten()(x)
    x = layers.Dense(256,activation='relu',use_bias=True)(x)

    z_mean = layers.Dense(latent_dim, name="z_mean")(x)
    z_log_var = layers.Dense(latent_dim, name="z_log_var")(x)
    z = vaes.Sampling()([z_mean, z_log_var])
    encoder = keras.Model(i, [z_mean, z_log_var, z], name="encoder")

    my_ml_lib.count_and_print_weights(encoder)

    return encoder

def architecture2(latent_dim):
    '''
        architecture based on decoder and ecoder of 
    '''
    img_input = keras.layers.Input(shape=(116, 12))
    a = keras.layers.Reshape((116, 12,1))(img_input)

    conv1 = keras.layers.Conv2D(256, (3, 3), activation='relu', padding='same')(a)
    conv1 = keras.layers.Conv2D(256, (3, 3), activation='relu', padding='same')(conv1)
    pool1 = keras.layers.MaxPooling2D((2, 2))(conv1)

    conv2 = keras.layers.Conv2D(256, (3, 3), activation='relu', padding='same')(pool1)
    conv2 = keras.layers.Conv2D(256, (3, 3), activation='relu', padding='same')(conv2)
    pool2 = keras.layers.MaxPooling2D((2, 2), padding='same')(conv2)

    conv3 = keras.layers.Conv2D(1, (3, 3), activation='relu', padding='same')(pool2)
    conv3 = keras.layers.Conv2D(1, (3, 3), activation='relu', padding='same')(conv3)

    x     = keras.layers.Flatten()(conv3)
    a     = keras.layers.Dense(87,activation = "tanh")(x)
    b     = keras.layers.Dense(87,activation = "tanh")(x)

    z     = vaes.Sampling()([a,b])

    encoder = keras.Model(img_input,[a,b,z],name = "encoder")

    my_ml_lib.count_and_print_weights(encoder)

    return encoder

