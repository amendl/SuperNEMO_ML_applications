'''
    Small library for some functions that are reused across this project

    author: adam.mendl@cvut.cz amend@hotmail.com
'''


import tensorflow as tf
from tensroflow import keras
import numpy as np
import ROOT
import argparse
import scikit-learn


def add_prefix(model, prefix: str, custom_objects=None):
    '''Adds a prefix to layers and model name while keeping the pre-trained weights for reusing loaded model for transfer learning
    Arguments:
        model: a tf.keras model
        prefix: a string that would be added to before each layer name
        custom_objects: if your model consists of custom layers you shoud add them pass them as a dictionary. 
            For more information read the following:
            https://keras.io/guides/serialization_and_saving/#custom-objects
    Returns:
        new_model: a tf.keras model having same weights as the input model.
    '''
    
    config = model.get_config()
    old_to_new = {}
    new_to_old = {}
    
    for layer in config['layers']:
        new_name = prefix + layer['name']
        old_to_new[layer['name']], new_to_old[new_name] = new_name, layer['name']
        layer['name'] = new_name
        layer['config']['name'] = new_name

        if len(layer['inbound_nodes']) > 0:
            for in_node in layer['inbound_nodes'][0]:
                in_node[0] = old_to_new[in_node[0]]
    
    for input_layer in config['input_layers']:
        input_layer[0] = old_to_new[input_layer[0]]
    
    for output_layer in config['output_layers']:
        output_layer[0] = old_to_new[output_layer[0]]
    
    config['name'] = prefix + config['name']
    new_model = tf.keras.Model().from_config(config, custom_objects)
    
    for layer in new_model.layers:
        layer.set_weights(model.get_layer(new_to_old[layer.name]).get_weights())
    
    return new_model



def count_and_print_weights(model,_print=True):
        trainable_count = np.sum([K.count_params(w) for w in model.trainable_weights])
        non_trainable_count = np.sum([K.count_params(w) for w in model.non_trainable_weights])
        if _print:
            print(f'Total params: {trainable_count + non_trainable_count}')
            print(f'Trainable params: {trainable_count}')
            print(f'Non-trainable params: {non_trainable_count}')

        return trainable_count,non_trainable_count


def process_command_line_arguments():
    '''
        returns:
            - device if --OneDeviceStategy otherwise None
            - devices if --MirroredStrategy otherwise None
    '''
    mode = 0
    device=None
    devices=[]
    for arg in sys.argv[1:]:
        if arg=="--OneDeviceStrategy":
            mode=1
        elif arg=="--MirroredStrategy":
            mode=2
        elif mode==1:
            device=arg
        elif mode==2:
            devices.append(arg)
        else:
            raise Exception(f"[Custom exception {__file__}:process_command_line_arguments]: \"{arg}\" is not valid parameter for this script.")
        
    return (device,devices)


def choose_strategy(device,devices=None):
    '''
        returns MirroredStrategy or OneDeviceStrategy
    '''
    if devices is not None and devices:
        return tf.distribute.MirroredStrategy(devices)
    elif device is not None:
        return tf.distribute.OneDeviceStrategy(device)
    else:
        raise Exception(f"[Custom exception {__file__}:strategy]: Not valid devices for tensorflow.distribute.MirroredStrategy nor valid device for tensorflow.distribute.OneDeviceStrategy were provided.")
   
def confusion(model, test_dataset,generatePdf=False):
    '''
        Calculates and prints confusion matrix
        TODO: generate directly pdf
    '''
    y_true = []

    for _,label in test_dataset:
        y_true.append(tf.argmax(label))

    print("Creating confusion matrix")
    prediction=model.predict(test_dataset.batch(1024))
    prediction = np.argmax(prediction, axis=1)
    cm = confusion_matrix(prediction, y_true)
    tf.print(cm,summarize=-1)
