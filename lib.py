#! /sps/nemo/scratch/amendl/AI/virtual_env_python391/bin/python


'''
    Small library with some functions that are reused across this project

    author: adam.mendl@cvut.cz amend@hotmail.com
'''


import tensorflow as tf
from tensorflow import keras
import tensorflow.keras.backend as K
import numpy as np
from sklearn.metrics import confusion_matrix, multilabel_confusion_matrix
import sys
import matplotlib.pyplot as plt
import inspect

def current_line():
    '''
        Returns line where this function was called. Used for printing error messages and raising Exceptions
    '''
    return inspect.currentframe().f_back.f_lineno

def plot_train_val_accuracy(history):
    
    plt.plot(history.history['accuracy'])
    plt.plot(history.history['val_accuracy'])
    plt.title('model accuracy')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(['train', 'val'], loc='upper left')
    plt.savefig(
        fname = 'training_accuracy.pdf',
        format = 'pdf'
    )


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
    if _print:
        print("Model summary:")
        model.summary()
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

def multilabel_confusion(model,test_dataset,i,generatePdf=False):
    '''
    
    '''
    print(f"Processing labels for {i} label classification")
    y_true = []
    for _,label in test_dataset:
        y_true.append(np.argpartition(label, -i)[-i:])
    print("Predicting test data")
    predictions=model.predict(test_dataset.batch(1024))
    print("Creating multilabel confusion matrix")
    cm = np.zeros((22,22))
    right_wrong = np.zeros(i+1)
    index = 0
    for predicted in predictions:
        y = y_true[index]
        x = np.argpartition(predicted, -i)[-i:]
        index1 = i-1

        while index1 > -1:
            index2 = len(x)-1
            while index2 > -1:
                if y[index1]==x[index2]:
                    y = np.delete(y,[index1],axis=0)
                    x = np.delete(x,[index2],axis=0)         
                    break         
                index2-=1
            index1-=1

        if len(x)==1 and len(y)==1:
            cm[x[0],y[0]]+=1
        right_wrong[len(x)]+=1
        print(len(x))

    tf.print(tf.constant(right_wrong)   ,summarize=-1)
    tf.print(tf.constant(cm)            ,summarize=-1) 



class RandomCell:
    '''
        
    '''
    def __init__(self,side,row,layer,rate,fire,distribution_,layer_function = None,row_function = None):
        '''
            Generates noise for specific cell
        '''
        if (self.fire != True and self.fire != False) or (side != 0 and side != 1) or row <0 or row>9 or layer<0 or layer >113 :
            raise Exception(f"Custom exception in {__file__}:{current_line()}: side = {side}, row = {row}, layer = {layer}, rate = {rate}, fire = {fire}")
        
        self.side   = side
        self.row    = row
        self.layer  = layer
        self.fire   = fire
        self.dist   = distribution_
        self.rate   = rate

        self.layer_function = lambda x: x if layer_function == None else layer_function
        self.row_function   = lambda x: x if row_function == None else row_function

    def __call__(self,top_projection,side_projection,front_projection,side):
        '''
        
        '''
        if (side==2 or side==self.side) and tf.random.uniform((1))[0] < self.rate:
            fill = 0. if self.fire == False else 1. 
            z = int((max(min(self.dist(),1490.),-1500.)+1500.)/100.)
            if top_projection!=None:
                top_projection[self.layer_function(self.layer),self.row_function(self.row)]         = fill
            if side_projection!=None:
                side_projection[z,self.row_function(self.row)]                                      = fill
            if front_projection!=None:
                front_projection[z,self.layer_function(self.layer)]                                 = fill

class RandomFullDetector():
    '''
        Generates noise for all detector
    '''
    def __init__(self,rate,fire,distribution_):
        '''
        '''
        self.rate = rate
        self.fire = fire
        self.dist = distribution_

    def __call__(self,top_projection,side_projection,front_projection,side):
        '''

        '''
        if not tf.random.uniform([]).numpy() < self.rate:
            fill    = 0. if self.fire == False else 1. 
            layer   = int(tf.random.uniform([],minval=0,maxval=9,dtype=tf.dtypes.int32))
            row     = int(tf.random.uniform([],minval=0,maxval=113,dtype=tf.dtypes.int32))
            z = int((max(min(self.dist(),1490.),-1500.)+1500.)/100.)
            top_projection[layer,row]           = fill
            side_projection[z,row]              = fill
            front_projection[z,layer]           = fill
            self(top_projection,side_projection,front_projection,side)

class ThresholdFinder:
    '''
        TODO general size of histogram
    '''
    def __init__(self):
        self.histo = np.zeros((100))
    def fill(self, original,truth,model_output):
        for i in range(100):
            threshold = 0.005 + float(i)*0.01
            for j in range(model_output.shape[0]):
                for k in range(model_output.shape[1]):
                    if original[j,k]>0.5:
                        if (model_output[j,k] > threshold) == (truth[j,k]>0.5):
                            self.histo[i]+=1
                        else:
                            self.histo[i]-=1
    def value(self):
        return 0.005+0.01*np.argmax(self.histo)
    def plot(self,**params):
        plt.plot(np.linspace(0.005,1.-0.005,num=100),self.histo)
        plt.axvline(0.005+0.01*float(np.argmax(self.histo)))
        plt.savefig(params)


class ParametersIterator:
    def __init__(self):
        pass

    def print_parameters(self):
        print(vars(self))

class AutoencoderOptions(ParametersIterator):
    '''
    
    '''

    def __init__(self):
        ParametersIterator.__init__(self)



class TrainingOptions(ParametersIterator):
    '''
    
    '''

    def __init__(self,tracks = [1,2],events_in_file=10000,files=[0,1,2,3,4,5,6,7],val_files = [8],test_files = [9],batch_size = 256,prefetch_size = 2):
        ParametersIterator.__init__(self)
        self.tracks             = tracks
        self.events_in_file     = events_in_file
        self.files              = files
        self.val_files          = val_files
        self.test_files         = test_files
        self.batch_size         = batch_size
        self.prefetch_size      = prefetch_size

    def get_shuffle_size(self) -> int:
        return len(self.tracks)*self.events_in_file*len(self.files)

    def approximate_steps_in_epoch(self) -> int:
        return int(self.get_shuffle_size()/self.batch_size)


if __name__=="__main__":
    raise NotImplementedError(f"[{__file__}:{current_line()-1}]: main is not implemented. This script should not be called directly.")