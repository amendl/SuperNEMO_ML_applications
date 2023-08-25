
import numpy as np

import tensorrt
import tensorflow as tf
from tensorflow import keras
import tensorflow.keras.backend as K

import matplotlib.pyplot as plt

import time
import sys
import datetime
import pickle

BUFFER_SIZE = 60000
BATCH_SIZE = 256
EPOCHS = 80
NOISE_DIM = 100


def import_arbitraty_module(module_name,path):
    import importlib.util
    import sys

    spec = importlib.util.spec_from_file_location(module_name,path)
    imported_module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = imported_module
    spec.loader.exec_module(imported_module)

    return imported_module

matteo    = import_arbitraty_module("matteo",               "/sps/nemo/scratch/amendl/AI/my_lib/latent_space_tricks/matteo.py")
task      = import_arbitraty_module("clustering_dataset",   "/sps/nemo/scratch/amendl/AI/my_lib/latent_space_tricks/clustering_dataset.py")
my_ml_lib = import_arbitraty_module("my_ml_lib",            "/sps/nemo/scratch/amendl/AI/my_lib/lib.py")


def make_generator_model():
    '''
    
    '''

    model = tf.keras.Sequential()
    model.add(keras.layers.Dense(7*7*256, use_bias=False, input_shape=(100,)))
    # model.add(keras.layers.BatchNormalization())
    model.add(keras.layers.LeakyReLU())

    model.add(keras.layers.Reshape((7, 7, 256)))
    assert model.output_shape == (None, 7, 7, 256)  # Note: None is the batch size

    model.add(keras.layers.Conv2DTranspose(128, (5, 5), strides=(1, 1), padding='same', use_bias=False))
    assert model.output_shape == (None, 7, 7, 128)
    # model.add(keras.layers.BatchNormalization())
    model.add(keras.layers.LeakyReLU())

    model.add(keras.layers.Conv2DTranspose(64, (5, 5), strides=(2, 2), padding='same', use_bias=False))
    assert model.output_shape == (None, 14, 14, 64)
    # model.add(keras.layers.BatchNormalization())
    model.add(keras.layers.LeakyReLU())

    model.add(keras.layers.Conv2DTranspose(1, (5, 5), strides=(2, 2), padding='same', use_bias=False, activation='tanh'))
    assert model.output_shape == (None, 28, 28, 1)

    return model


def make_discriminator_model():
    '''
    
    '''

    model = tf.keras.Sequential()
    model.add(keras.layers.Conv2D(64, (5, 5), strides=(2, 2), padding='same',
                                     input_shape=[28, 28, 1]))
    model.add(keras.layers.LeakyReLU())
    model.add(keras.layers.Dropout(0.3))

    model.add(keras.layers.Conv2D(128, (5, 5), strides=(2, 2), padding='same'))
    model.add(keras.layers.LeakyReLU())
    model.add(keras.layers.Dropout(0.3))

    model.add(keras.layers.Flatten())
    model.add(keras.layers.Dense(1))

    return model








cross_entropy = tf.keras.losses.BinaryCrossentropy(from_logits=True)

class GAN_Options(my_ml_lib.AutoencoderOptions,my_ml_lib.TrainingOptions):
    '''
    
    '''
    def __init__(self,tracks = [1,2],events_in_file=10000,files=[0,1,2,3,4,5,6,7,8],val_files = [9],test_files = [],discriminator_access_truth = True,batch_size = 256, prefetch_size = 2):
        my_ml_lib.AutoencoderOptions.__init__(self)
        my_ml_lib.TrainingOptions.__init__(self,tracks,events_in_file,files,val_files,test_files,batch_size,prefetch_size)

        self.discriminator_access_truth = discriminator_access_truth

class GAN_History:
    '''
    
    '''
    def __init__(self,x,loss_discriminator_true,loss_discriminator_fake,loss_generator,epoch_boundaries,eval_loss_discriminator_true,eval_loss_discriminator_fake,eval_loss_generator,real_training_accuracy,fake_training_accuracy,real_val_accuracy,fake_val_accuracy):
        self.x                              = x
        self.loss_discriminator_true        = loss_discriminator_true
        self.loss_discriminaotr_fake        = loss_discriminator_fake
        self.loss_generator                 = loss_generator
        self.epoch_boundaries               = epoch_boundaries

        self.eval_loss_discriminator_true   = eval_loss_discriminator_true
        self.eval_loss_discriminaotr_fake   = eval_loss_discriminator_fake
        self.eval_loss_generator            = eval_loss_generator

        self.real_training_accuracy         = real_training_accuracy
        self.fake_training_accuracy         = fake_training_accuracy

        self.real_val_accuracy              = real_val_accuracy
        self.fake_val_accuracy              = fake_val_accuracy
    
    def __getstate__(self):
        return self.__dict__
    
    def __setstate__(self, d):
        self.__dict__ = d

    def plot(self,plot_losses = 'losses.pdf', plot_accuracies = 'accuracies.pdf'):
        if plot_losses:
            plt.clf()
            plt.plot(self.x,self.loss_discriminator_true,label='loss discriminator (true)')
            plt.plot(self.x,self.loss_discriminaotr_fake,label='loss discriminator (fake)')
            plt.plot(self.x,self.loss_generator,label='loss generator')
            plt.scatter(self.epoch_boundaries,self.eval_loss_discriminator_true,label='eval loss discriminator (true)')
            plt.scatter(self.epoch_boundaries,self.eval_loss_discriminaotr_fake,label='eval loss discriminator (fake)')
            plt.scatter(self.epoch_boundaries,self.eval_loss_generator,label='eval loss generator')
            plt.legend()
            plt.savefig(plot_losses)
        if plot_accuracies:
            plt.clf()
            plt.plot(self.x,self.real_training_accuracy, label="dis real accuracy")
            plt.plot(self.x,self.fake_training_accuracy, label="dis fake accuracy")
            plt.scatter(self.epoch_boundaries,self.real_val_accuracy, label="dis real accuracy")
            plt.scatter(self.epoch_boundaries,self.fake_val_accuracy, label="dis fake accuracy")
            plt.legend()
            plt.savefig(plot_accuracies) 


def loss_for_generator(real,fake):
    return cross_entropy(tf.ones_like(fake), fake)

def loss_for_discriminator(real, fake):
    real_loss = cross_entropy(tf.ones_like(real), real)
    fake_loss = cross_entropy(tf.zeros_like(fake), fake)
    return real_loss + fake_loss,real_loss, fake_loss

@tf.function
def process_batch(batch,noise,training,generator_network,discriminator_network) -> tuple:

    # def to_channels(tensors):
    #     for i in range(len(tensors)):
    #         tensors[i] = tf.cast(tensors[i],dtype=tf.float64)
    #         if tf.rank(tensors[i])==3:
    #             tensors[i] = tf.expand_dims(tensors[i],axis=3)
    #     return tf.concat(tensors,axis=3)
    
    generated = generator_network(noise,training = training)

    return generated, discriminator_network(tf.cast(batch,dtype=tf.float64),training = training), discriminator_network(tf.cast(generated,dtype=tf.float64),training = training)

@tf.function
def train_step(dataset_batch,generator_network,discriminator_network,accuracy_real,accuracy_fake,generator_optimizer = None,discriminator_optimizer = None,training = False) -> tuple:
    '''
    
    '''
    noise = tf.random.normal([BATCH_SIZE, NOISE_DIM])


    gen_tape = tf.GradientTape()
    dis_tape = tf.GradientTape()

    try:
        if training:
            gen_tape.__enter__()
            dis_tape.__enter__()
        
        generated, real_discriminator_output,fake_discriminator_output = process_batch(dataset_batch,noise,training,generator_network,discriminator_network)

        dis_loss,rl, fl = loss_for_discriminator   (real_discriminator_output,fake_discriminator_output)
        gen_loss        = loss_for_generator       (real_discriminator_output,fake_discriminator_output)
        # dis_loss,rl,fl,gen_loss = (0.,0.,0.,0.)
    finally:
        if gen_tape._recording:
            gen_tape.__exit__(None,None,None)
        elif training:
            raise RuntimeError("Fatal Error: Tape 'gen_tape' should be in recording state, but it is not! Training step failed!")
        if dis_tape._recording:
            dis_tape.__exit__(None,None,None)
        elif training:
            raise RuntimeError("Fatal Error: Tape 'dis_tape' should be in recording state, but it is not! Training step failed!")



    if training and generator_optimizer is not None and discriminator_optimizer is not None:
        gradients_of_generator      = gen_tape.gradient(gen_loss, generator_network.trainable_variables)
        gradients_of_discriminator  = dis_tape.gradient(dis_loss, discriminator_network.trainable_variables)

        generator_optimizer         .apply_gradients(grads_and_vars = zip(gradients_of_generator, generator_network.trainable_variables))
        discriminator_optimizer     .apply_gradients(grads_and_vars = zip(gradients_of_discriminator, discriminator_network.trainable_variables))
    
    return (
        {
            "discriminator full"    : dis_loss,
            "discriminator real"    : rl,
            "discriminator fake"    : fl,
            "generator loss"        : gen_loss
        },
        (
            generated,
            real_discriminator_output,
            fake_discriminator_output
        )
    )

def run_test(test_dataset,generator_network,discriminator_network,means,accuracy_real,accuracy_fake):
    NUMBER_OF_CLASSES = 2
    confusion = np.zeros((NUMBER_OF_CLASSES,NUMBER_OF_CLASSES))

    map(lambda x: x.reset_state(),means)
    accuracy_real.reset_state()
    accuracy_fake.reset_state()
        
    # run evaluation
    for test_dataset_batch in test_dataset:
        (dic, (generated, _, _ )) = train_step(test_dataset_batch,generator_network,discriminator_network,accuracy_real,accuracy_fake)
        for mean_holder, mean_value in zip(means,dic.values()):
            mean_holder.update_state(mean_value)

    # print results
    for name,mean_holder in zip(["discriminator full","discriminator real","discriminator fake","generator loss"],means):
        print(name,str(mean_holder.result().numpy()).ljust(10),sep=" : ",end="; ")

    return confusion


def print_discriminator_accuracies(real_accuracy_metric,fake_accuracy_metric,sep="; ",end=""):

    print(f"discriminator_real_accuracy : {real_accuracy_metric if real_accuracy_metric is float else real_accuracy_metric.result().numpy()}",f"discriminator_fake_accuracy : {fake_accuracy_metric if fake_accuracy_metric is float else fake_accuracy_metric.result().numpy()}",sep=sep,end=end)



def train(epochs,dataset,val_dataset,generator_optimizer,discriminator_optimizer,generator_network,discriminator_network,plot_every = None):
    '''
    
    '''
    # if epochs is not int:
    #     raise TypeError("Argument 'epochs' should be integer")
    # if epochs < 1:
    #     raise ValueError("Argument 'epochs' should be positive integer")
    # if not issubclass(type(generator_optimizer),tf.optimizers.Optimizer):
    #     raise TypeError("Argument 'generator_optimizer' should be subclass of 'tf.optimizers.Optimizer'")
    # if not issubclass(type(discriminator_optimizer),tf.optimizers.Optimizer):
    #     raise TypeError("Argument 'discriminator_optimizer' should be subclass of 'tf.optimizers.Optimizer'")
    # if (plot_every is not None) or (plot_every is not int):
    #     raise TypeError("Argument 'plot_every' should be integer or None")
    # if plot_every is int and plot_every < 1:
    #     raise ValueError("Argument 'plot_every' should be positive integer.")
    
    # TODO checking instances of datasets is probably too complicated    
    

    x                            = []
    epoch_boundaries             = []
    arrays                       = [[]for _ in range(4)]
    means                        = [tf.keras.metrics.Mean() for _ in range(4)]

    accuracy_real_training       = tf.keras.metrics.BinaryAccuracy()
    accuracy_fake_training       = tf.keras.metrics.BinaryAccuracy()

    accuracy_fake_val            = tf.keras.metrics.BinaryAccuracy()
    accuracy_real_val            = tf.keras.metrics.BinaryAccuracy()

    eval_loss_discriminator_true = []
    eval_loss_discriminator_fake = []
    eval_loss_generator          = []

    real_training_accuracy       = []
    fake_training_accuracy       = []

    real_val_accuracy            = []
    fake_val_accuracy            = []

    steps = None
    for epoch_index in range(1,epochs+1):
        print(f'\n[train function]: Epoch #{epoch_index} started:')
        sys.stdout.flush()
        start = time.time()

        # Training
        i = 1
        for dataset_batch in dataset:
            print(f"[{str(i).rjust(7)} from {'Unknown' if steps == None else str(steps).rjust(7)}",end="]: ")
            # reset states of discriminator accuracies
            accuracy_real_training.reset_state()
            accuracy_fake_training.reset_state()
            # run testing step print metrics and add data to training history
            for j,(metric_key,metric_value) in enumerate(train_step(dataset_batch,generator_network,discriminator_network,accuracy_real_training,accuracy_fake_training,generator_optimizer,discriminator_optimizer,True)[0].items()):
                print(metric_key,str(metric_value.numpy()).ljust(10),sep=" : ",end="; ")
                if plot_every is not None and i%plot_every == 0:
                    x.append(i+(steps if steps is not None else 0)*(epoch_index-1))
                    arrays[j].append(metric_value.numpy())
            print_discriminator_accuracies(accuracy_real_training,accuracy_fake_training)
            real_training_accuracy.append(accuracy_real_training.result().numpy)
            fake_training_accuracy.append(accuracy_fake_training.result().numpy)

            i+=1
            print()
            sys.stdout.flush()
        steps = i-1

        epoch_boundaries.append((epoch_index-1)*steps)

        

        print(f'[train function]: Epoch #{epoch_index} took {int(time.time() - start)} s')
        sys.stdout.flush()

    return GAN_History(
        x                            = x,
        loss_discriminator_true      = arrays[1],
        loss_discriminator_fake      = arrays[2],
        loss_generator               = arrays[3],
        epoch_boundaries             = epoch_boundaries,
        eval_loss_discriminator_true = eval_loss_discriminator_true,
        eval_loss_discriminator_fake = eval_loss_discriminator_fake,
        eval_loss_generator          = eval_loss_generator,
        real_training_accuracy       = real_training_accuracy,
        fake_training_accuracy       = fake_training_accuracy,
        real_val_accuracy            = real_val_accuracy,
        fake_val_accuracy            = fake_val_accuracy   
    )

if __name__=="__main__":

    options = GAN_Options(
        tracks                          = [2],
        events_in_file                  = 19000,
        files                           = [0,1,2,3,4,5,6,7,8],
        val_files                       = [9],
        test_files                      = [],
        discriminator_access_truth      = True,
        batch_size                      = BATCH_SIZE,
        prefetch_size                   = 2
    )
    print("\n====================")
    print("Run parameters:")
    options.print_parameters()
    print("\n====================")
    print("Physical devices:")
    print(tf.config.list_physical_devices())
    device,devices  = my_ml_lib.process_command_line_arguments()
    strategy        = my_ml_lib.choose_strategy(device,devices)
    print("\n====================")
    print("Running with strategy: ",str(strategy))
    try:
        print(" on device ", strategy.device)
    except:
        pass
    try:
        print(" on devices ",strategy.devices)
    except:
        pass
    sys.stdout.flush()


    with strategy.scope():
        generator = make_generator_model()
        print("\n====================")
        print("Generator summary:")
        my_ml_lib.count_and_print_weights(generator,True)

        discriminator = make_discriminator_model()

        print("\n====================")
        print("Discriminator summary:")
        my_ml_lib.count_and_print_weights(generator,True)

        (train_images, train_labels), (_, _) = tf.keras.datasets.mnist.load_data()

        train_images = train_images.reshape(train_images.shape[0], 28, 28, 1).astype('float32')
        train_images = (train_images - 127.5) / 127.5  # Normalize the images to [-1, 1]
        train_dataset = tf.data.Dataset.from_tensor_slices(train_images).shuffle(BUFFER_SIZE).batch(BATCH_SIZE).prefetch(1)

    history = train(
        epochs                      = EPOCHS,
        dataset                     = train_dataset,
        val_dataset                 = None,
        generator_optimizer         = tf.optimizers.Adam(),
        discriminator_optimizer     = tf.optimizers.Adam(),
        generator_network           = generator,
        discriminator_network       = discriminator,
        plot_every                  = 20
    )

    with open('history.pickle','wb') as f:
        pickle.dump(history,f)
    with open('options.pickle','wb') as f:
        pickle.dump(options,f)


    generator           .save("generator")
    discriminator       .save("discriminator")


