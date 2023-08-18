
import numpy as np

import tensorflow as tf
from tensorflow import keras
import tensorflow.keras.backend as K

import matplotlib.pyplot as plt

import time
import sys
import datetime
import pickle


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

    def plot(self,plot_losses = True, plor_accuracies = True):
        raise NotImplementedError()




def discriminator_architecture(gan_options):
    img_input1 = keras.layers.Input(shape=(116, 12))
    img_input2 = keras.layers.Input(shape=(116, 12))
    img_input1 = keras.layers.Reshape((116,12,1))(img_input1)
    img_input2 = keras.layers.Reshape((116,12,1))(img_input2)

    conv1 = keras.layers.Conv2D(32, (3, 3), activation='relu', padding='same')(keras.layers.Concatenate()((img_input1,img_input2)))
    conv1 = keras.layers.Dropout(0.2)(conv1)
    conv1 = keras.layers.Conv2D(32, (3, 3), activation='relu', padding='same')(conv1)
    pool1 = keras.layers.MaxPooling2D((2, 2))(conv1)

    conv2 = keras.layers.Conv2D(64, (3, 3), activation='relu', padding='same')(pool1)
    conv2 = keras.layers.Dropout(0.2)(conv2)
    conv2 = keras.layers.Conv2D(64, (3, 3), activation='relu', padding='same')(conv2)
    pool2 = keras.layers.MaxPooling2D((2, 2), padding='same')(conv2)

    conv3 = keras.layers.Conv2D(128, (3, 3), activation='relu', padding='same')(pool2)
    conv3 = keras.layers.Dropout(0.2)(conv3)
    conv3 = keras.layers.Conv2D(128, (3, 3), activation='relu', padding='same')(conv3)

    x     = keras.layers.Flatten()(conv3)
    x     = keras.layers.Dense(256,activation='tanh')(x)
    x     = keras.layers.Dense(128,activation='tanh')(x)
    x     = keras.layers.Dense(1)(x) # XXX uses from_logits=True so no sigmoid

    retval = keras.Model(inputs = [img_input1,img_input2],outputs = x)

    print("\n====================")
    print("Discriminator summary:")
    my_ml_lib.count_and_print_weights(retval,True)

    return retval


def loss_for_generator(real,fake):
    return cross_entropy(tf.ones_like(fake), fake)

def loss_for_discriminator(real, fake):
    real_loss = cross_entropy(tf.ones_like(real), real)
    fake_loss = cross_entropy(tf.zeros_like(fake), fake)
    return real_loss + fake_loss,real_loss, fake_loss

@tf.function
def process_batch(batch,training,generator_network,discriminator_network) -> tuple:

    # def to_channels(tensors):
    #     for i in range(len(tensors)):
    #         tensors[i] = tf.cast(tensors[i],dtype=tf.float64)
    #         if tf.rank(tensors[i])==3:
    #             tensors[i] = tf.expand_dims(tensors[i],axis=3)
    #     return tf.concat(tensors,axis=3)
    
    generated = generator_network(batch[0],training = training)

    return generated, discriminator_network((tf.cast(batch[0],dtype=tf.float64),tf.cast(batch[1],dtype=tf.float64)),training = training), discriminator_network((tf.cast(batch[0],dtype=tf.float64),tf.cast(generated,dtype=tf.float64)),training = training)

@tf.function
def train_step(dataset_batch,generator_network,discriminator_network,accuracy_real,accuracy_fake,generator_optimizer = None,discriminator_optimizer = None,training = False) -> tuple:
    '''
    
    '''
    gen_tape = tf.GradientTape()
    dis_tape = tf.GradientTape()

    try:
        if training:
            gen_tape.__enter__()
            dis_tape.__enter__()
        
        generated, real_discriminator_output,fake_discriminator_output = process_batch(dataset_batch,training,generator_network,discriminator_network)

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

    # add to accuracy
    accuracy_real.update_state()


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

    map(lambda x: x.reset_state(),means)
    accuracy_real.reset_state()
    accuracy_fake.reset_state()
        
    # run evaluation
    for test_dataset_batch in test_dataset:
        (dic, (generated, _, _ )) = train_step(test_dataset_batch,generator_network,discriminator_network,accuracy_real,accuracy_fake)
        for mean_holder, mean_value in zip(means,dic.values()):
            mean_holder.update_state(mean_value)
        # TODO add analysis of usefullness of clustering
    
    # print results
    for name,mean_holder in zip(["discriminator full","discriminator real","discriminator fake","generator loss"],means):
        print(name,str(mean_holder.result().numpy()).ljust(10),sep=" : ",end="; ")


def print_discriminator_accuracies(real_accuracy_metric,fake_accuracy_metric,sep="; ",end=""):
    print(f"discriminator_real_accuracy : {real_accuracy_metric.result().numpy() if real_accuracy_metric is tf.keras.metrics.BinaryAccuracy else real_accuracy_metric}",f"discriminator_fake_accuracy : {fake_accuracy_metric.result().numpy() if fake_accuracy_metric is tf.keras.metrics.BinaryAccuracy else fake_accuracy_metric}",sep=sep,end=end)




def train(epochs,dataset,val_dataset,generator_optimizer,discriminator_optimizer,generator_network,discriminator_network,plot_every = None):
    '''
    
    '''
    if epochs is not int:
        raise TypeError("Argument 'epochs' should be integer")
    if epochs < 1:
        raise ValueError("Argument 'epochs' should be positive integer")
    if not issubclass(type(generator_optimizer),tf.optimizers.Optimizer):
        raise TypeError("Argument 'generator_optimizer' should be subclass of 'tf.optimizers.Optimizer'")
    if not issubclass(type(discriminator_optimizer),tf.optimizers.Optimizer):
        raise TypeError("Argument 'discriminator_optimizer' should be subclass of 'tf.optimizers.Optimizer'")
    if (plot_every is not None) or (plot_every is not int):
        raise TypeError("Argument 'plot_every' should be integer or None")
    if plot_every is int and plot_every < 1:
        raise ValueError("Argument 'plot_every' should be positive integer.")
    
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

        print("\nValidation results: ",end="")
        
        # run validation

        run_test(val_dataset,generator_network,discriminator_network,means,accuracy_real_val,accuracy_fake_val)
        eval_loss_discriminator_true.append(means[1].result().numpy())
        eval_loss_discriminator_fake.append(means[2].result().numpy())
        eval_loss_generator         .append(means[3].result().numpy())
        print_discriminator_accuracies(accuracy_real_val,accuracy_fake_val)
        real_val_accuracy.append(accuracy_real_val.result().numpy)
        fake_val_accuracy.append(accuracy_fake_val.result().numpy)

        print()

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
        batch_size                      = 256,
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
        generator = matteo.with_skip_connection(options)
        print("\n====================")
        print("Generator summary:")
        my_ml_lib.count_and_print_weights(generator,True)

        discriminator = discriminator_architecture(options)

        dataset = tf.data.Dataset.from_generator(
            generator = lambda: task.generator(options.tracks,options.files,options.events_in_file),
            output_signature=(tf.TensorSpec(shape=(3),dtype=tf.int32))
        )
        dataset = dataset.shuffle(options.get_shuffle_size(),reshuffle_each_iteration = True).map(task.load_event)

        val_dataset = tf.data.Dataset.from_generator(
            generator = lambda: task.generator(options.tracks,options.val_files,options.events_in_file),
            output_signature=(tf.TensorSpec(shape=(3),dtype=tf.int32))
        )
        val_dataset = val_dataset.map(task.load_event)

    history = train(
        epochs                      = 30,
        dataset                     = dataset.batch(options.batch_size).prefetch(options.prefetch_size),
        val_dataset                 = val_dataset.batch(options.batch_size).prefetch(options.prefetch_size),
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


