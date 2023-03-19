
import tensorflow as tf
import numpy as np
import ROOT

import sys

def generator(number_of_tracks,files,events):
    '''
    
    '''
    for tracks in range(1,number_of_tracks+1):
        for file_idx in files:
            for event_idx in range(events):
                yield tf.constant([tracks,file_idx,event_idx],dtype=tf.int64)

def py_load_event(event_id):
    tracker_image = np.zeros((9,113))

    file = ROOT.TFile("../../datasets/haystack/%i/%ilines_%i.root" % (event_id[0],event_id[0],event_id[1]))
    tree = file.Get("hit_tree")

    tree.GetEntry(event_id.numpy()[2])
    
    for i in range(tree.grid_layer.size()):
        # if tree.grid_side[i]==1: 
        tracker_image[tree.grid_layer[i],tree.grid_column[i]] = 1. # CHANGE

    return tf.constant(tracker_image)

def py_load_event_height(event_id):
    tracker_image = np.zeros((30,113))

    file = ROOT.TFile("../../datasets/haystack/%i/%ilines_%i.root" % (event_id[0],event_id[0],event_id[1]))
    tree = file.Get('hit_tree')

    tree.GetEntry(event_id.numpy()[2])

    for i in range(tree.grid_layer.size()):
        height_index = int((max(min(tree.wirez[i],1490.),-1500.)+1500.)/100.)
        tracker_image[height_index,tree.grid_column[i]] = 1. # CHANGE
    
    return tf.constant(tracker_image)

def py_load_event_front(event_id):
    tracker_image = np.zeros((30,9))

    file = ROOT.TFile("../../datasets/haystack/%i/%ilines_%i.root" % (event_id[0],event_id[0],event_id[1]))
    tree = file.Get('hit_tree')

    tree.GetEntry(event_id.numpy()[2])

    for i in range(tree.grid_layer.size()):
        height_index = int((max(min(tree.wirez[i],1490.),-1500.)+1500.)/100.)
        tracker_image[height_index,tree.grid_layer[i]] = 1. # CHANGE

    return tf.constant(tracker_image)

def py_load_result(event_id):
    '''
    
    '''
    result = np.zeros((4))
    result[event_id[0]-1]=1.
    return tf.constant(result)

# @tf.function # XXX Doesn't work with decorator
def load_event(event_id):
    '''
        Calls python code inside py_load_event_representation
    '''

    # [event,]            = tf.py_function(func=py_load_event,            inp=[event_id], Tout=[tf.TensorSpec(shape=(9,113),   dtype=tf.float64)])
    # event.              set_shape((9,113))
    # [event_height,]     = tf.py_function(func=py_load_event_height,     inp=[event_id], Tout=[tf.TensorSpec(shape=(30,113),  dtype=tf.float64)])
    # event_height.       set_shape((30,113))
    [event_front,]      = tf.py_function(func=py_load_event_front,      inp=[event_id], Tout=[tf.TensorSpec(shape=(30,9),    dtype=tf.float64)])
    event_front.set_shape((30,9))
    [result,]           = tf.py_function(func=py_load_result,           inp=[event_id], Tout=[tf.TensorSpec(shape=(4),       dtype=tf.float64)])
    result.set_shape((4))

    return event_front,result

if __name__ == '__main__':
    event, _, _, _ = load_event((2,9,16))
    print(event)
    tf.print(event,summarize=-1)