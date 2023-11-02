

import numpy as np

import tensorrt
import tensorflow as tf
import ROOT

import sys

def generator(number_of_tracks,files,events):
    '''
    
    '''
    for tracks in number_of_tracks:
        for file_idx in files:
            for event_idx in range(events):
                yield tf.constant([tracks,file_idx,event_idx],dtype=tf.int64)

def py_load_event(tree):
    tracker_image = np.zeros((9,113))
    
    for i in range(tree.grid_layer.size()):
        # if tree.grid_side[i]==1: 
        tracker_image[tree.grid_layer[i],tree.grid_column[i]] = 1. # CHANGE

    return tracker_image

def py_load_event_height(tree):
    tracker_image = np.zeros((30,113))

    for i in range(tree.grid_layer.size()):
        height_index = int((max(min(tree.wirez[i],1490.),-1500.)+1500.)/100.)
        tracker_image[height_index,tree.grid_column[i]] = 1. # CHANGE
    
    return tracker_image

def py_load_event_front(tree):
    tracker_image = np.zeros((30,9))


    for i in range(tree.grid_layer.size()):
        height_index = int((max(min(tree.wirez[i],1490.),-1500.)+1500.)/100.)
        tracker_image[height_index,tree.grid_layer[i]] = 1. # CHANGE

    return tracker_image

def py_load_result(tree,event_id):
    '''
    
    '''
    
    result = np.zeros((1))
    result[0]=1 if event_id.numpy()[0]>3 else 0
    return tf.constant(result)

def open_root_file(event_id):
    file = ROOT.TFile("my_generator_t%i_%i.root" % (event_id[0],event_id[1]))
    tree = file.Get('hit_tree')

    tree.GetEntry(event_id.numpy()[2])

    return tree

# @tf.function # XXX Doesn't work with decorator
def load_event_helper(event_id,noise = []):
    '''
        Calls python code inside
    '''

    file = ROOT.TFile("my_generator_t%i_%i.root" % (event_id[0],event_id[1])) if event_id[0] < 4 else ROOT.TFile("my_generator_t%i_%i_signal_like.root" % (event_id[0]-3,event_id[1]))
    tree = file.Get('hit_tree')

    tree.GetEntry(event_id.numpy()[2])

    event  = py_load_event(tree)
    result = py_load_result(tree,event_id)

    # for n in noise:
    #     n(event,event_height,event_front,2)


    return tf.constant(event),result

def load_event(event_id,noise=[]):
    '''
        If you want to change which data should be used for training, use return value of this function
    '''
    [event_top,result]    =   tf.py_function(func=load_event_helper,inp=[event_id,noise],Tout=[tf.TensorSpec(shape=(9,113),    dtype=tf.float64),tf.TensorSpec(shape=(1),        dtype=tf.float64)])

             
    event_top           .set_shape((9,113))
    result              .set_shape((1))
    return event_top,result

def load_event_testing(event_id,noise=[]):
    '''
        If you want to change which data should be used for training, use return value of this function
    '''
    [event_top,result]    =   tf.py_function(func=load_event_helper,inp=[event_id,noise],Tout=[tf.TensorSpec(shape=(9,113),    dtype=tf.float64),tf.TensorSpec(shape=(1),        dtype=tf.float64)])

             
    event_top           .set_shape((9,113))
    result              .set_shape((1))
    return event_top

def print_group(group):
    if type(group) is tuple or type(group) is list :
        for i in group:
            tf.print(i,summarize=-1)
    else:
        tf.print(group,summarize=-1)

def print_event(event_number):
    '''
        load event should return (event_top,event_height,event_front),(result,result_column,result_row)
    '''
    print(event_number)
    event_group, result_group = load_event(event_number)
    print_group(result_group)
    print_group(event_group)

if __name__ == '__main__':
    # print_event((1,0,14))
    # print_event((1,0,15))
    print_event((2,0,16))
    print_event((2,0,17))
    # print_event((3,0,18))
    # print_event((3,0,19))
    print_event((4,0,20))
    print_event((4,0,21))

