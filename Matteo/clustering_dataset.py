'''
    TODO not finished
'''



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

def build_image(tree,start_array,end_array):
    '''
    
    '''
    tracker_image = np.zeros((116,12))


    for s,e in zip(start_array,end_array):
        for i in range(s,e):
            tracker_image[tree.grid_column[i]+2,tree.grid_layer[i]+2] = 1.

    return tracker_image

def load_event_helper(event_id):
    '''
    
    '''

    file = ROOT.TFile("../../../datasets/my_generator_with_hint/%i/my_generator_t%i_%i.root" % (event_id[0],event_id[0],event_id[1]))
    tree = file.Get('hit_tree')
    tree.GetEntry(event_id.numpy()[2])
    
    smallest = 1000000
    smallest_index = 0
    # find first
    for i in range(tree.grid_column.size()):
        # print(tree.grid_column[i])
        if tree.grid_column[i]<smallest:
            smallest = tree.grid_column[i]
            smallest_index = i

    # find first group
    smallest_group = -1
    for i in range(tree.track_split.size()):
        if smallest_index < tree.track_split[i]:
            smallest_group = i
            break
    
    image1 = tf.constant(build_image(tree,[0],[tree.grid_column.size()]))
    image2 = None
    if smallest_group == 0:
        image2 = build_image(tree,[tree.track_split[smallest_group]+1],[tree.grid_column.size()])
    elif smallest_group == event_id[0]-1:
        image2 = build_image(tree,[0],[tree.track_split[smallest_group-1]])
    else:
        image2 = build_image(tree,[0,tree.track_split[smallest_group]+1],[tree.grid_column.size()])
    
    return image1,tf.constant(image2)

def load_event(event_id):
    [event1,event2] = tf.py_function(
        func=load_event_helper,
        inp=[event_id],
        Tout=[
            tf.TensorSpec(shape=(116,12),    dtype=tf.float64),
            tf.TensorSpec(shape=(116,12),    dtype=tf.float64)
        ]
    )
    event1.set_shape((116,12))
    event2.set_shape((116,12))
    return event1,event2

if __name__ == '__main__':
    for i in range(0,100):
        tf.print(load_event_helper(tf.constant([2,1,i])),summarize=-1)
        input("")

    