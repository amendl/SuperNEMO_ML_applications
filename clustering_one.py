'''



'''

# general libraries
from typing import Final
import os
import sys
import time
#tensorflow related
import tensorflow as tf
import numpy as np
# import discriminator
from discriminator import front_view as discriminator_architecture
# import generator
from generator import front_view as generator_architecture
# dataset loading
import my_lib.datasets.clustering_one as task


# This method returns a helper function to compute cross entropy loss
cross_entropy = tf.keras.losses.BinaryCrossentropy(from_logits=True)
# BUG should we use softmax before Crossentropy? Probably not woth BinaryCrossentropy

# Create generator and discriminator models
discriminator_model = discriminator_architecture()
generator_model     = generator_architecture()

# Define optimizers
generator_optimizer = tf.keras.optimizers.Adam(1e-4)
discriminator_optimizer = tf.keras.optimizers.Adam(1e-4)

EPOCHS                      : Final[int] = 30
EPOCH_CHECKPOINT_START      : Final[int] = 3
EPOCH_CHECKPOINT_FREQUENCY  : Final[int] = 5
CHECKPOINT_PREFIX           : Final[str] = "ckpt"
CHECKPOINT = tf.train.Checkpoint(
    generator_optimizer         = generator_optimizer,
    discriminator_optimizer     = discriminator_optimizer,
    generator                   = generator_model,
    discriminator               = discriminator_model
)

USE_BADLY_CLUSTERED         : Final[bool] = False

EVENTS_FROM_FILE            : Final[int] = 10000
TRACKS                      : Final[int] = 4

CLUSTERED_WELL_FILES        : Final[{int}] = [0,1,2,3,4,5,6,7] # TODO
CLUSTERED_WELL_SIZE         : Final[int] = 0 # TODO
CLUSTERED_BADLY_FILES       : Final[{int}] = [0,1,2,3,4,5,6,7] # TODO
CLUSTERED_BADLY_SIZE        : Final[int] = 0 # TODO
NOT_CLUSTERED_FILES         : Final[{int}] = [0,1,2,3,4,5,6,7] # TODO
NOT_CLUSTERED_SIZE          : Final[int] = 0 # TODO

BATCH_SIZE                  : Final[int] = 1024
BATCHES_TO_LOG              : Final[int] = 10



def discriminator_loss(real_output, fake_output):
    '''
    forcintg discriminator to identify:
        * real output as ones and
        * generated data as zero
    '''
    real_loss = cross_entropy(tf.ones_like(real_output),real_output)
    fake_loss = cross_entropy(tf.zeros_like(fake_output), fake_output)
    return real_loss + fake_loss

def generator_loss(fake_output):
    '''
        forcing generator to generate real data samples that will be recognized as real data
    '''
    return cross_entropy(tf.ones_like(fake_output), fake_output)


@tf.function
def train_step(clustered_data,not_clustered_data):
    '''
        GAN training step. 
    '''
    with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
        clustered_tracks    = generator_model(not_clustered_data,training = True)
        # TODO how to put together channels into discriminator
        real_output         = discriminator_model(clustered_data,training = True)
        fake_output         = discriminator_model(clustered_tracks, training = True)

        gen_loss            = generator_loss(fake_output)
        disc_loss           = discriminator_loss(real_output,fake_output)

        print('gen_loss: %f;disc_loss: %f'%(gen_loss,disc_loss))

    gradients_of_generator          = gen_tape.     gradient(gen_loss, generator_model.trainable_variables)
    gradients_of_discriminator      = disc_tape.    gradient(disc_loss, discriminator_model.trainable_variables)

    generator_optimizer.        apply_gradients(zip(gradients_of_generator, generator_model.trainable_variables))
    discriminator_optimizer.    apply_gradients(zip(gradients_of_discriminator, discriminator_model.trainable_variables))

def train(clustered_dataset_well,not_clustered_dataset,clustered_dataset_badly=None):
    for epoch in range(epochs):
        epoch_start = time.time()
        batch_counter = 0

# https://medium.com/ymedialabs-innovation/how-to-use-dataset-and-iterators-in-tensorflow-with-code-samples-3bb98b6b74ab




                        
if __name__ == '__main__':
    '''
        Entry point for script
    '''

    # well clustered datasets
    clustered_dataset_well = tf.data.Dataset.from_generator(
        generator = lambda: task.generator_well(TRACKS,CLUSTERED_WELL_FILES,EVENTS_FROM_FILE),
        output_signature=(tf.TensorSpec(shape=(3),dtype=tf.int64))
    )
    clustered_dataset_well = clustered_dataset_well.shuffle(CLUSTERED_WELL_SIZE,reshuffle_each_iteration=True)
    clustered_dataset_well.map(task.load_and_cluster_well)

    # badly clustered datasets
    if USE_BADLY_CLUSTERED:
        clustered_dataset_badly = tf.data.Dataset.from_generator(
            generator = lambda: task.generator_badly(TRACKS,CLUSTERED_BADLY_FILES,EVENTS_FROM_FILE), 
            output_signature=(tf.TensorSpec(shape=(6),dtype=tf.int64))
        )
        clustered_dataset_badly = clustered_dataset_badly.shuffle(CLUSTERED_BADLY_SIZE,reshuffle_each_iteration=True)
        clustered_dataset_badly.map(task.load_and_cluster_badly)

    # not clustered datasets
    not_clustered_dataset = tf.data.Dataset.from_generator(
        generator = lambda: task.generator_not_clustered(TRACKS,NOT_CLUSTERED_FILES,EVENTS_FROM_FILE),
        output_signature=(tf.TensorSpec(shape=(6),dtype=tf.int64))
    )
    not_clustered_dataset = not_clustered_dataset.shuffle(NOT_CLUSTERED_SIZE,reshuffle_each_iteration=True)
    not_clustered_dataset.map(task.load_and_cluster_well)


    # training loop
    for epoch in range(EPOCHS):
        start = time.time()

        for image_batch in dataset:
            # run train step
            pass

        if epoch >= EPOCH_CHECKPOINT_START and (epoch - EPOCH_CHECKPOINT_START) % EPOCH_CHECKPOINT_FREQUENCY == 0:
            CHECKPOINT.save(file_prefix = CHECKPOINT_PREFIX)


    discriminator_model.save("discriminator")
    generator_model.save("generator")



