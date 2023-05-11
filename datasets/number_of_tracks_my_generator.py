
import numpy as np

import tensorflow as tf
import ROOT

import sys

def generator(number_of_tracks,files,events):
    '''
    
    '''
    for tracks in range(1,number_of_tracks+1):
        for file_idx in files:
            for event_idx in range(events):
                yield tf.constant([tracks,file_idx,event_idx],dtype=tf.int64)

def py_load_event(tree):
    tracker_image = np.zeros((9,113))
    
    for i in range(tree.grid_layer.size()):
        # if tree.grid_side[i]==1: 
        tracker_image[tree.grid_layer[i],tree.grid_column[i]] = 1. # CHANGE

    return tf.constant(tracker_image)

def py_load_event_height(tree):
    tracker_image = np.zeros((30,113))

    for i in range(tree.grid_layer.size()):
        height_index = int((max(min(tree.wirez[i],1490.),-1500.)+1500.)/100.)
        tracker_image[height_index,tree.grid_column[i]] = 1. # CHANGE
    
    return tf.constant(tracker_image)

def py_load_event_front(tree):
    tracker_image = np.zeros((30,9))


    for i in range(tree.grid_layer.size()):
        height_index = int((max(min(tree.wirez[i],1490.),-1500.)+1500.)/100.)
        tracker_image[height_index,tree.grid_layer[i]] = 1. # CHANGE

    return tf.constant(tracker_image)

def py_load_result(tree,event_id):
    '''
    
    '''
    calo_row    = np.zeros(15)
    calo_column = np.zeros(22)


    for row_hit in tree.calo_row:
        calo_row[row_hit]=1.
    for column_hit in tree.calo_column:
        calo_column[column_hit]=1.

    result = np.zeros((4))
    result[event_id[0]-1]=1.
    return tf.constant(result),tf.constant(calo_row),tf.constant(calo_column)

def open_root_file(event_id):
    file = ROOT.TFile("../../datasets/my_generator/%i/my_generator_t%i_%i.root" % (event_id[0],event_id[0],event_id[1]))
    tree = file.Get('hit_tree')

    tree.GetEntry(event_id.numpy()[2])

    return tree

# @tf.function # XXX Doesn't work with decorator
def load_event_helper(event_id):
    '''
        Calls python code inside py_load_event_representation
    '''

    file = ROOT.TFile("../../datasets/my_generator/%i/my_generator_t%i_%i.root" % (event_id[0],event_id[0],event_id[1]))
    tree = file.Get('hit_tree')

    tree.GetEntry(event_id.numpy()[2])

    event                                  = py_load_event(tree)           
    # [event_height,]                             = tf.py_function(func=py_load_event_height,     inp=[tree], Tout=[tf.TensorSpec(shape=(30,113),  dtype=tf.float64)])
    # event_height                                .set_shape((30,113))
    # [event_front,]                              = tf.py_function(func=py_load_event_front,      inp=[tree], Tout=[tf.TensorSpec(shape=(30,9),    dtype=tf.float64)])
    # event_front                                 .set_shape((30,9))
    result,result_row,result_column          = py_load_result(tree,event_id)


    return event,result,result_row,result_column

def load_event(event_id):
    [event,result,_,result_column]=tf.py_function(func=load_event_helper,inp=[event_id],Tout=[
                                                                                tf.TensorSpec(shape=(9,113),   dtype=tf.float64),
                                                                            
                                                                                tf.TensorSpec(shape=(4),dtype=tf.float64),
                                                                                tf.TensorSpec(shape=(15),dtype=tf.float64),
                                                                                tf.TensorSpec(shape=(22),dtype=tf.float64)
                                                                                
                                                                             ]
                                                                            )
    event                                       .set_shape((9,113))
    result                                      .set_shape((4))
    result_column                               .set_shape((22))

    return event,(result,result_column)

def print_event(event_number):
    print(event_number)
    event_group, result_group = load_event(event_number)
    event,_,_= event_group
    result,result_row,result_column=result_group
    tf.print(result,            summarize=-1)
    tf.print(result_row,        summarize=-1)
    tf.print(result_column,     summarize=-1)
    tf.print(event,             summarize=-1)


if __name__ == '__main__':
    print_event((1,0,14))
    print_event((1,0,15))
    print_event((1,0,16))
    print_event((1,0,17))
    print_event((1,0,18))
    print_event((1,0,19))
    print_event((1,0,20))

