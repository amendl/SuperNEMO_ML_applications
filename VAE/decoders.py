
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
    i = keras.Input(shape=(latent_dim,))
    x = layers.Dense(64)(i)
    x = layers.Dense(29*3*64,activation="relu")(x)
    x = layers.Reshape((29,3,64))(x)
    conv3 = keras.layers.Conv2D(64, (3, 3), activation='relu', padding='same')(x)
    conv3 = keras.layers.Dropout(0.2)(conv3)
    conv3 = keras.layers.Conv2D(64, (3, 3), activation='relu', padding='same')(conv3)

    up1 = keras.layers.UpSampling2D((2, 2))(conv3)
    conv4 = keras.layers.Conv2D(64, (3, 3), activation='relu', padding='same')(up1)
    conv4 = keras.layers.Dropout(0.2)(conv4)
    conv4 = keras.layers.Conv2D(64, (3, 3), activation='relu', padding='same')(conv4)

    up2 = keras.layers.UpSampling2D((2, 2))(conv4)
    conv5 = keras.layers.Conv2D(32, (3, 3), activation='relu', padding='same')(up2)
    conv5 = keras.layers.Dropout(0.2)(conv5)
    conv5 = keras.layers.Conv2D(32, (3, 3), activation='relu', padding='same')(conv5)

    out = keras.layers.Conv2D(1, 3, padding='same')(keras.layers.UpSampling2D(1)(conv5)) # (2,3)
    model = keras.models.Model(i, out, name="decoder")

    my_ml_lib.count_and_print_weights(model)

    return model