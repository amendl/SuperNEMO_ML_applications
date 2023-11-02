
import tensorflow as tf
import numpy as np
import ROOT

ITH_TRACK_OUTPUT = 2

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
    tracker_image_true = np.zeros((116,12))
    tracker_image_false = np.ones((116,12))

    for s,e in zip(start_array,end_array):
        for i in range(s,e):
            tracker_image_true[tree.grid_column[i]+2,tree.grid_layer[i]+2] = 1.
            tracker_image_false[tree.grid_column[i]+2,tree.grid_layer[i]+2] = 0.

    return tracker_image_true,tracker_image_false

def load_event_helper(event_id):
    '''
    
    '''

    file = ROOT.TFile("my_generator_t%i_%i.root" % (event_id[0],event_id[1]))
    tree = file.Get('hit_tree')
    tree.GetEntry(event_id.numpy()[2])

    
    
    smallest_values = [[1000000,i] for i in range(event_id[0])]
    index = 0
    for current_track_index in range(event_id[0]):
        while index < tree.track_split[current_track_index]:
            if smallest_values[current_track_index][0] > tree.grid_column[index]:
                smallest_values[current_track_index][0] = tree.grid_column[index]
            index += 1

    smallest_values.sort(key=lambda x:x[0])

    
    image_true, image_false = build_image(tree,[0],[tree.grid_column.size()])
    image2 = None
    smallest_group = smallest_values[ITH_TRACK_OUTPUT-1][1]
    if smallest_group == 0:
        image2 = build_image(tree,[tree.track_split[smallest_group]+1],[tree.grid_column.size()])[0]
    elif smallest_group == event_id[0]-1:
        image2 = build_image(tree,[0],[tree.track_split[smallest_group-1]])[0]
    else:
        image2 = build_image(tree,[0,tree.track_split[smallest_group]+1],[tree.track_split[smallest_group-1],tree.grid_column.size()])[0]
    
    return tf.constant(image_true),tf.constant(image2)

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
        a,b = load_event_helper(tf.constant([4,1,i]))
        tf.print(tf.constant(a.numpy().T),summarize=-1)
        tf.print(tf.constant(b.numpy().T),summarize=-1)
        input("")

    