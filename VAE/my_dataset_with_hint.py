#! /sps/nemo/scratch/amendl/AI/virtual_env_python391/bin/python


import tensorflow as tf
import numpy as np
import ROOT


def generator(number_of_tracks,files,events):
    '''
    
    '''
    for tracks in number_of_tracks:
        for file_idx in files:
            for event_idx in range(events):
                yield tf.constant([tracks,file_idx,event_idx],dtype=tf.int64)

def load_event_helper(event_id):
    tracker_image1 = np.zeros((9,113))
    tracker_image2 = np.zeros((116,12))



    file = ROOT.TFile("../../../datasets/my_generator_with_hint/%i/my_generator_t%i_%i.root" % (event_id[0],event_id[0],event_id[1]))
    tree = file.Get('hit_tree')
    tree.GetEntry(event_id.numpy()[2])

    for i in range(tree.grid_layer.size()):
        tracker_image1[tree.grid_layer[i],tree.grid_column[i]] = 1.
        tracker_image2[tree.grid_column[i]+2,tree.grid_layer[i]+2] = 1.

    return tf.constant(tracker_image1),tf.constant(tracker_image2)

def load_event(event_id):
    [event1,event2] = tf.py_function(
        func=load_event_helper,
        inp=[event_id],
        Tout=[
            tf.TensorSpec(shape=(9,113),    dtype=tf.float64),
            tf.TensorSpec(shape=(116,12),    dtype=tf.float64)
        ]
    )
    event1.set_shape((9,113))
    event2.set_shape((116,12))
    return event1,event2
