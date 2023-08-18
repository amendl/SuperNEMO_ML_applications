#! /sps/nemo/scratch/amendl/AI/virtual_env_python391/bin/python


import tensorflow as tf
from tensorflow import keras
from keras import layers


class Sampling(layers.Layer):
    '''
        Sampling layer for the reparametrisation trick
    '''
    def call(self,inputs):
        z_mean, z_log_var = inputs
        batch = tf.shape(z_mean)[0]
        dim = tf.shape(z_mean)[1]
        epsilon = tf.random.normal(shape=(batch, dim))
        return z_mean + tf.exp(0.5 * z_log_var) * epsilon

class VAE(keras.Model):
    '''
    
    '''
    def __init__(self, encoder, decoder, **kwargs):
        super().__init__(**kwargs)
        self.encoder = encoder
        self.decoder = decoder
        self.total_loss_tracker = keras.metrics.Mean(name="total_loss")
        self.reconstruction_loss_tracker = keras.metrics.Mean(name="reconstruction_loss")
        self.kl_loss_tracker = keras.metrics.Mean(name="kl_loss")

    @property
    def metrics(self):
        return [
            self.total_loss_tracker,
            self.reconstruction_loss_tracker,
            self.kl_loss_tracker,
        ]

    def train_step(self, data):
        kl_loss = 0
        reconstruction_loss = 0
        loss = 0
        with tf.GradientTape() as tape:
            # run data through neural network
            z_mean, z_log_var, z = self.encoder(data[0])
            reconstruction = self.decoder(z)
            print(tf.shape(data[1]))
            # get losses
            reconstruction_loss = tf.reduce_mean(
                tf.reduce_sum(
                    keras.losses.binary_crossentropy(tf.expand_dims(data[1],axis=3), reconstruction), axis=(1, 2)
                )
            )
            kl_loss = -0.5 * (1 + z_log_var - tf.square(z_mean) - tf.exp(z_log_var))
            kl_loss = tf.reduce_mean(tf.reduce_sum(kl_loss, axis=1))
            loss = reconstruction_loss + kl_loss

        # apply gradients
        grads = tape.gradient(loss, self.trainable_weights)
        self.optimizer.apply_gradients(zip(grads, self.trainable_weights))
        
        self.total_loss_tracker             .reset_state()
        self.reconstruction_loss_tracker    .reset_state()
        self.kl_loss_tracker                .reset_state()

        # save losses
        self.total_loss_tracker             .update_state(loss)
        self.reconstruction_loss_tracker    .update_state(reconstruction_loss)
        self.kl_loss_tracker                .update_state(kl_loss)

        # return losses
        return {
            "loss": loss,
            "reconstruction_loss": reconstruction_loss,
            "kl_loss": kl_loss
        }
    
    def call(self,inputs):
        return self.decoder(self.encoder(inputs)[2])


