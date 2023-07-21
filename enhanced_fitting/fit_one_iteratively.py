import ROOT

ROOT.gSystem.Load('/sps/nemo/scratch/amendl/AI/real_testing/TKEventEdited/TKEvent/lib/libTKEvent.so')


import tensorflow as tf
from tensorflow import keras
import numpy as np




def create_histo(name,label=None):
    if label == None:
        label = name
    return ROOT.TH1F(name,label,50,0.,0.25)

def are_trhits_same(trhit1,trhit2):
    return trhit1.get_SRL('r') == trhit2.get_SRL('r') and trhit1.get_SRL('l') == trhit2.get_SRL('l')

def is_trhit_broken(trhit):
    return trhit.get_r()>35.0 or trhit.get_r()==-1. 

def build_input_for_ml(event,filtering=None):
    
    hits = None
    if filtering is not None:
        hits = filtering(event.get_tr_hits())
    else:
        hits = event.get_tr_hits()

    event_top                              = np.zeros((9,113))
    event_height                           = np.zeros((30,113))
    event_front                            = np.zeros((30,9))

    for tracker_hit in hits:
        if not is_trhit_broken(tracker_hit):
            height          = tracker_hit.get_h()
            height_index    = int((max(min(height,1490.),-1500.)+1500.)/100.)
            layer           = tracker_hit.get_SRL('l')
            row             = tracker_hit.get_SRL('r')
            event_top[layer,row]            = 1.
            event_height[height_index,row]  = 1.
            event_front[height_index,layer] = 1.

    return tf.expand_dims(tf.constant(event_top),axis=0),tf.expand_dims(tf.constant(event_height),axis=0),tf.expand_dims(tf.constant(event_front),axis=0)



def filter_on_neighbours(hits):
    retval = []
    for hit in hits:
        if is_trhit_broken(hit):
            retval.append(hit)
            continue
        adjacent = 0
        for hit2 in hits:
            if abs(hit.get_SRL('r')-hit2.get_SRL('r'))<2 and abs(hit.get_SRL('l')-hit2.get_SRL('l')) < 2 and abs(hit.get_SRL('s')-hit2.get_SRL('s')) ==0 and (not is_trhit_broken(hit2)):
                adjacent+=1
        if adjacent > 1:
            retval.append(hit)
    return retval

if __name__ == "__main__":

    ordered_size    = 6
    print_each      = 10
    start           = 0
    end             = 70
    model_dir       = '/sps/nemo/scratch/amendl/AI/my_lib/combined_model2/model'
    input_file      = '/sps/nemo/scratch/amendl/AI/real_testing/runs/Run-974.root'
    output_file     = 'results_from_hits.root'

    

    model = keras.models.load_model(model_dir)
    print("model from folder ", model_dir, "loaded") 


    full_histogram      = create_histo("full")
    ordered_histograms  = [create_histo("%i_best" % (i+1)) for i in range(ordered_size)]

    file = ROOT.TFile(input_file,"READ")
    tree = file.Get("Event")
    print("file ", input_file, "opened")

    for i in range(start,end):
        tree.GetEntry(i)
        event = tree.Eventdata
        event.set_r("Manchester","distance")

        hits = event.get_tr_hits()

        ml_input = build_input_for_ml(event,filter_on_neighbours)
        predicted = np.argmax(model(ml_input)+1)
        print("evet # ", i, "prediction: ",predicted+1)


        for _ in range(predicted+1):
            event.reconstruct_multi_from_hits(hits,False)
            if len(event.get_tracks())==0:
                break
            last_track = event.get_tracks().back()
            
            hits_len = len(hits)
            index = 0
            while index < hits_len:
                if hits[index] in event.get_tr_hits():
                    hits.erase(hits.begin()+index)
                else:
                    index+=1
                hits_len = len(hits)
            
        event.make_top_projection("Events_visu_filtered/%i_predicted_%i.png"%(i,predicted+1))

        likelihoods = [track.get_confidence() for track in event.get_tracks()]
        for l in likelihoods:
            full_histogram.Fill(l)
        likelihoods.sort(reverse=True)
        for count,l in enumerate(likelihoods):
            if count >= ordered_size:
                break
            ordered_histograms[count].Fill(l)
    file.Close()
    print("file ",input_file, " closed")

    file = ROOT.TFile(output_file,"RECREATE")
    print("file ", output_file, "opened")
    full_histogram.Write()
    for h in ordered_histograms:
        h.Write()
    file.Close
    print("file ",output_file, "closed")
