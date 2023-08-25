#! /sps/nemo/scratch/amendl/AI/virtual_env_python391/bin/python

import tensorrt
import tensorflow as tf
from tensorflow import keras
import matplotlib.pyplot as plt

NOISE_DIM = 100

if __name__=="__main__":
    model = keras.models.load_model("generator",compile=False)

    predictions = model(tf.random.normal([16, NOISE_DIM]), training=False)

    fig = plt.figure(figsize=(4, 4))

    for i in range(predictions.shape[0]):
        plt.subplot(4, 4, i+1)
        plt.imshow(predictions[i, :, :, 0] * 127.5 + 127.5, cmap='gray')
        plt.axis('off')
    
    plt.savefig('image.png')

    plt.clf()

    (train_images, train_labels), (_, _) = tf.keras.datasets.mnist.load_data()

    train_images = train_images.reshape(train_images.shape[0], 28, 28, 1).astype('float32')
    train_images = (train_images - 127.5) / 127.5  # Normalize the images to [-1, 1]


    fig = plt.figure(figsize=(4, 4))


    for i in range(16):
        plt.subplot(4, 4, i+1)
        plt.imshow(train_images[i] * 127.5 + 127.5, cmap='gray')
        plt.axis('off')

    
    plt.savefig('image_original.png')

