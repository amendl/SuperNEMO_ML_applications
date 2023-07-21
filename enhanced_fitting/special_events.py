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

def build_input_for_ml(tracker_hits):

    event_top                              = np.zeros((9,113))
    event_height                           = np.zeros((30,113))
    event_front                            = np.zeros((30,9))

    for tracker_hit in tracker_hits:
        if not is_trhit_broken(tracker_hit):
            height          = tracker_hit.get_h()
            height_index    = int((max(min(height,1490.),-1500.)+1500.)/100.)
            layer           = tracker_hit.get_SRL('l')
            row             = tracker_hit.get_SRL('r')
            event_top[layer,row]            = 1.
            event_height[height_index,row]  = 1.
            event_front[height_index,layer] = 1.

    return tf.expand_dims(tf.constant(event_top),axis=0),tf.expand_dims(tf.constant(event_height),axis=0),tf.expand_dims(tf.constant(event_front),axis=0)


def try_event(event_no,tree,remove,model):
    tree.GetEntry(event_no)
    event = tree.Eventdata
    event.set_r("Manchester","distance")
    ml_input = build_input_for_ml(event.get_tr_hits())
    predicted = np.argmax(model(ml_input)+1)

    new_tr_hits = event.get_tr_hits()
    index = 0
    while index < len(new_tr_hits):
        is_present = False
        for r in remove:
            if are_trhits_same(new_tr_hits[index],r):
                is_present = True
                break
        if is_present:
            new_tr_hits.erase(new_tr_hits.begin()+index)
        else:
            index+=1

    event.make_top_projection("Events_visu/%i_original_%i.png"%(event_no,predicted+1))
    

    drawing = ROOT.TKEvent(event.get_run_number(),event.get_event_number())
    for e in new_tr_hits:
        if not is_trhit_broken(e):
            drawing.add_tracker_hit(ROOT.TKtrhit(e))

    ml_input = build_input_for_ml(drawing.get_tr_hits())
    predicted2 = np.argmax(model(ml_input)+1)
    drawing.make_top_projection("Events_visu/%i_edited_%i.png"%(event_no,predicted2+1))

def tkprint(event_no,tree):
    print(event_no,":")
    tree.GetEntry(event_no)
    event = tree.Eventdata
    for hit in event.get_tr_hits():
        if not is_trhit_broken(hit):
            print(hit.get_SRL('s'),hit.get_SRL('r'),hit.get_SRL('l'))



def trhit(S,R,L):
    return ROOT.TKtrhit(np.array([S,R,L],dtype=np.int32),np.zeros((7),dtype=np.int64))

if __name__ == "__main__":
    model_dir       = '/sps/nemo/scratch/amendl/AI/my_lib/combined_model2/model'
    input_file      = '/sps/nemo/scratch/amendl/AI/real_testing/runs/Run-974.root'

    model = keras.models.load_model(model_dir)
    print("model from folder ", model_dir, "loaded") 

    file = ROOT.TFile(input_file,"READ")
    tree = file.Get("Event")
    print("file ", input_file, "opened")

    def t(a,b):
        try_event(a,tree,b,model)

    # t(1,[trhit(1,55,2)])
    # t(3,[trhit(0,111,7)])
    # t(9,[trhit(0,22,0)])
    # t(10,[trhit(1,10,0)])
    # t(11,[trhit(0,13,4),trhit(0,9,3)])
    # t(17,[trhit(0,2,6),trhit(0,111,7)])
    # t(20,[trhit(1,10,0)])
    # t(23,[trhit(0,111,7)])
    # t(39,[trhit(1,9,7),trhit(0,2,6)])
    # t(51,[trhit(1,15,2)])
    # t(52,[trhit(0,2,6),trhit(1,82,3)])

    tkprint(52,tree)

    file.Close()
    print("file ",input_file, " closed")