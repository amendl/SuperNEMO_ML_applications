import datasets.number_of_tracks as task

from sklearn.metrics import confusion_matrix

import numpy as np

import tensorflow as tf
from tensorflow import keras
from keras import layers # Dense, Droupout, Softmax, BatchNormalization, Conv2D
from keras.callbacks import EarlyStopping

import matplotlib.pyplot as plt

def load_trained_model(file, index):
    A = keras.models.load_model(file)
    for layer in A.layers:
        layer.trainable=False
    print(A.summary())
    return (A,A.layers[index])

if __name__=='__main__':
    A, A_model_output = load_trained_model("../big_model/model",-5)
    B, B_model_output = load_trained_model("../height_model2/model",-5)
    C, C_model_output = load_trained_model("../model_front/model",-5)
    exit(0)
    layers.Concatenate()([A_model_output,B_model_output,C_model_output])


    model = keras.Model(inputs=[A.input, B.input,C.input], outputs=x)

    model.compile(
        optimizer = 'adam',
        loss = tf.keras.losses.CategoricalCrossentropy(),
        metrics = ['accuracy']
    )
    for epochs in range(2):
        history = model.fit(
            x = dataset.batch(256),
            epochs = 15,
            validation_data = val_dataset.batch(256),
            callbacks=[EarlyStopping(monitor='val_accuracy',mode='max',baseline=0.9,start_from_epoch=5,min_delta=0.01)]
        )

    model.compile(
        optimizer = 'adam',
        loss = tf.keras.losses.CategoricalCrossentropy(),
        metrics = ['accuracy']
    )

    for layer in A.layers:
        layer.trainable=True
    for layer in B.layers:
        layer.trainable=True
    for layer in C.layers:
        layer.trainable=True





    print(A.layers[-3])
    # B = keras.model.load_model("../height_model/model")

    # concat = layers.Concatenate()([A.layers[].output,B.layers[].output])

    # x = layers.Dense(64,activation='relu',use_bias=True)(x)
    # x = layers.Dense(4)(x)
    # x = layers.Softmax()(x)

