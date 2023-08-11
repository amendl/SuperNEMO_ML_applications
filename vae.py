#! /sps/nemo/scratch/amendl/AI/virtual_env_python391/bin/python


import tensorflow as tf
from tensorflow import keras
from keras import layers

import sys

def import_arbitrary_module(module_name,path):
    import importlib.util
    import sys

    spec = importlib.util.spec_from_file_location(module_name,path)
    imported_module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = imported_module
    spec.loader.exec_module(imported_module)

    return imported_module

vaes        = import_arbitrary_module("vaes",           "/sps/nemo/scratch/amendl/AI/my_lib/latent_space_tricks/VAE/lib.py")
vae_encoder = import_arbitrary_module("vae_encoder",    "/sps/nemo/scratch/amendl/AI/my_lib/latent_space_tricks/VAE/encoders.py")
vae_decoder = import_arbitrary_module("vae_encoder",    "/sps/nemo/scratch/amendl/AI/my_lib/latent_space_tricks/VAE/decoders.py")
task        = import_arbitrary_module("task",           "/sps/nemo/scratch/amendl/AI/my_lib/latent_space_tricks/VAE/my_dataset_with_hint.py")
my_ml_lib   = import_arbitrary_module("my_ml_lib",      "/sps/nemo/scratch/amendl/AI/my_lib/lib.py")



if __name__=="__main__":
    tracks          = 2
    files           = 10
    events          = 5000
    dataset_size    = tracks*files*events
    latent_size     = 3

    print(tf.config.list_physical_devices())

    device,devices  = my_ml_lib.process_command_line_arguments()
    strategy        = my_ml_lib.choose_strategy(device,devices)

    print("Running with strategy: ",str(strategy), " ",end="")
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
        encoder   = vae_encoder.architecture1(latent_size)
        decoder   = vae_decoder.architecture1(latent_size)
        model     = vaes.VAE(encoder,decoder)

        model.compile(optimizer=keras.optimizers.Adam())

        dataset = tf.data.Dataset.from_generator(
            generator = lambda: task.generator([1,2],[0,1,2,3,4,5,6,7],events),
            output_signature=(tf.TensorSpec(shape=(3),dtype=tf.int32))
        )
        dataset = dataset.map(task.load_event).shuffle(dataset_size,reshuffle_each_iteration = True)

        val_dataset = tf.data.Dataset.from_generator(
            generator = lambda: task.generator([1,2],[8],events),
            output_signature=(tf.TensorSpec(shape=(3),dtype=tf.int32))
        )
        val_dataset = val_dataset.map(task.load_event)

        # test_dataset = tf.data.Dataset.from_generator(
        #     generator = lambda: task.generator([1,2],[9],events),
        #     output_signature=(tf.TensorSpec(shape=(3),dtype=tf.int32))
        # )
        # test_dataset = test_dataset.map(task.load_event)


    history = model.fit(
        x = dataset.batch(64).prefetch(1),
        epochs = 5,
        validation_data = val_dataset.batch(64).prefetch(1)
    )

    model.save('model')

    my_ml_lib.plot_train_val_accuracy(history)















