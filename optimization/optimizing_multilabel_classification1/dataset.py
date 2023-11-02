
import tensorflow as tf
import numpy as np
import ROOT

MAX_NUMBER_OF_TRACKS = 3

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


def inject_image(tree,start_array,end_array,nparray):
    for s,e in zip(start_array,end_array):
        for i in range(s,e):
            nparray[tree.grid_column[i]+2,tree.grid_layer[i]+2] = 1.



def inject_smallest_group(tree,nparray,event_id,smallest_group):
    starts = None
    ends   = None

    if smallest_group == 0:
        starts = [0]
    else:
        starts = [tree.track_split[smallest_group-1]+1]
    ends = [tree.track_split[smallest_group]]

    inject_image(tree,starts,ends,nparray)




    # if smallest_group == 0:
    #     inject_image(tree,[tree.track_split[smallest_group]+1],[tree.grid_column.size()],nparray)
    # elif smallest_group == event_id[0]-1:
    #     inject_image(tree,[0],[tree.track_split[smallest_group-1]],nparray)
    # else:
    #     inject_image(tree,[0,tree.track_split[smallest_group]+1],[tree.grid_column.size()],nparray)


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
    ground_truth = np.zeros((116,12,MAX_NUMBER_OF_TRACKS))

    # ground_truth = np.expand_dims(image_false, axis=2)
    counter = 0
    mid = (len(smallest_values)) // 2
    for (_,smallest1), (_,smallest2) in zip(smallest_values[:mid], smallest_values[::-1]):
        inject_smallest_group(tree,ground_truth[:,:,counter],event_id,smallest1)
        counter+=1
        inject_smallest_group(tree,ground_truth[:,:,counter],event_id,smallest2)
        counter+=1

    if len(smallest_values)% 2 == 1:
        inject_smallest_group(tree,ground_truth[:,:,counter],event_id,smallest_values[mid][1])


    # for _,smallest_group in smallest_values:
    #     image2 = None
    #     if smallest_group == 0:
    #         image2 = build_image(tree,[tree.track_split[smallest_group]+1],[tree.grid_column.size()])[0]
    #     elif smallest_group == event_id[0]-1:
    #         image2 = build_image(tree,[0],[tree.track_split[smallest_group-1]])[0]
    #     else:
    #         image2 = build_image(tree,[0,tree.track_split[smallest_group]+1],[tree.grid_column.size()])[0]
    #     ground_truth = np.concatenate((ground_truth, np.expand_dims(image2, axis=2)),axis=2)
    #     counter += 1
    # while counter < MAX_NUMBER_OF_TRACKS:
    #     ground_truth = np.concatenate((ground_truth,np.zeros((116,12,1))),axis=2)

    
    return tf.constant(image_true),tf.constant(ground_truth)

def load_event(event_id):
    [event1,event2] = tf.py_function(
        func=load_event_helper,
        inp=[event_id],
        Tout=[
            tf.TensorSpec(shape=(116,12),    dtype=tf.float64),
            tf.TensorSpec(shape=(116,12,MAX_NUMBER_OF_TRACKS),    dtype=tf.float64)
        ]
    )
    event1.set_shape((116,12))
    event2.set_shape((116,12,MAX_NUMBER_OF_TRACKS))
    return event1,event2

def nice_print(a):
    for i in range(MAX_NUMBER_OF_TRACKS):
        tf.print(tf.constant(a[:,:,i].T),summarize=-1)
    print()

if __name__ == '__main__':
    for i in range(0,100):
        print([MAX_NUMBER_OF_TRACKS,1,i])
        a,b = load_event_helper(tf.constant([MAX_NUMBER_OF_TRACKS,1,i]))  # i%MAX_NUMBER_OF_TRACKS+1
        tf.print(tf.constant(a.numpy().T),summarize=-1)
        print("Result:")
        nice_print(b.numpy())
        input("")

    