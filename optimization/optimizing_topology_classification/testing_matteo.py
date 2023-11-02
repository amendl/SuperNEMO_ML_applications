#! /sps/nemo/scratch/amendl/AI/virtual_env_python391/bin/python
import tensorrt
import tensorflow as tf
from tensorflow import keras
import numpy as np
import sys
import tensorflow.keras.layers


import ROOT
import matplotlib.pyplot as plt


from keras import backend as K
import os

import math


def import_arbitrary_module(module_name,path):
    import importlib.util
    import sys

    spec = importlib.util.spec_from_file_location(module_name,path)
    imported_module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = imported_module
    spec.loader.exec_module(imported_module)

    return imported_module

task  = import_arbitrary_module("task","/sps/nemo/scratch/amendl/AI/my_lib/optimizing_topology_classification/scripts/dataset.py")
attention_lib   = import_arbitrary_module("attention",'/sps/nemo/scratch/amendl/AI/my_lib/attention/cbam.py')
my_ml_lib       = import_arbitrary_module("my_ml_lib",'/sps/nemo/scratch/amendl/AI/my_lib/lib.py')



def push_attention_block(option_name,layer,use_normalization = False):
    ratio = int(os.environ[option_name])
    if ratio == 0:
        return layer
    else:
        if use_normalization:
            return keras.layers.concatenate([keras.layers.LayerNormalization(axis=[1,2,3])(attention_lib.cbam_block(layer,ratio)),layer],axis=-1)
        else:
            return keras.layers.concatenate([attention_lib.cbam_block(layer,ratio),layer],axis=-1)


def architecture_with_normalization():
    '''
    
    '''
    i = keras.Input(shape=(9,113))
    img = keras.layers.Reshape((9,113,1))(i)
    x = keras.layers.Conv2D(256,(3,15),activation = 'relu',padding="same")(img)
    x = keras.layers.MaxPooling2D(pool_size = (1,2),strides=(1,2))(push_attention_block("P1",x,True))
    x = keras.layers.Conv2D(128,(3,7),activation = 'relu',padding="same")(x)
    x = keras.layers.MaxPooling2D(pool_size = (1,2),strides=(1,2))(push_attention_block("P2",x,True))
    x = keras.layers.Conv2D(128,(3,3),activation = 'relu',padding="same")(x)
    x = keras.layers.MaxPooling2D(pool_size = (2,2))(push_attention_block("P3",x,True))
    x = keras.layers.Conv2D(128,(3,3),activation = 'relu',padding="same")(x) 
    x = keras.layers.MaxPooling2D(pool_size = (2,2))(push_attention_block("P4",x,True))
    x = keras.layers.Conv2D(128,(3,3),activation = 'relu',padding="same")(x) 
    x = keras.layers.MaxPooling2D(pool_size = (2,2))(push_attention_block("P5",x,True)) 
    x = keras.layers.Dropout(0.2)(x)
    x = keras.layers.Flatten()(x)
    x = keras.layers.Dense(256,activation='relu',use_bias=True)(x)
    x = keras.layers.Dense(128,activation='relu',use_bias=True)(x)
    x = keras.layers.Dense(1,activation='sigmoid')(x)

    model = keras.Model(inputs = i, outputs = x)

    return model


def tf_to_numpy(tensor,matteo_shape=True):
    def non_vectorized_f(x):
        if x>0.5:
            return 1.
        else:
            return 0.
    vectorized_f = np.vectorize(non_vectorized_f)

    dataset = tf.data.Dataset.from_tensor_slices(tensor)
    iterator = dataset.as_numpy_iterator()
    elements = [element for element in iterator]
    x = np.vstack(elements)
    if matteo_shape:
        x = np.transpose(x)

        x = np.delete(x, (0,1,11), axis=0)
        x = np.delete(x,(0,1), axis=1)
    # x = vectorized_f(x)
    return x

def my_print(tensor):
    # print(tensor.shape)
    for i in range(116):
        print(chr(9608),end="")
    print()
    for i in range(tensor.shape[0]):
        print(chr(9608),end="")
        for j in range(tensor.shape[1]):
            if tensor[i,j]>0.545:
                print(chr(0x2299),end="")
            else:
                print(" ",end="")
        print(chr(9608))
    for i in range(116):
        print(chr(9608),end="")


def draw_event(event_id,model):
    original,truth  = task.load_event_helper(event_id)
    model_output    = model(tf.reshape(original,[1,116,12]))
    original        = tf_to_numpy(tf.reshape(original,[116,12]))
    model_output    = tf_to_numpy(tf.reshape(model_output,[116,12]))

    e = ROOT.TKEvent()
    e_truth = ROOT.TKEvent()
    for i in range(model_output.shape[0]):
        for j in range(model_output.shape[1]):
            if model_output[i,j]>0.5:
                e.add_tracker_hit(np.array([0,j,i],dtype=np.int32),np.array([10,20000,30000,40000,50000,6000,7000],dtype=np.int_))
            if original[i,j]>0.545:
                e_truth.add_tracker_hit(np.array([0,j,i],dtype=np.int32),np.array([10,20000,30000,40000,50000,6000,7000],dtype=np.int_))
    
    e.make_top_projection(f"WentWrong/event_{event_id}_generated.pdf")
    e_truth.make_top_projection(f"WentWrong/event_{event_id}_original.pdf")


def analyse_event(ID,model,fillers):
    original,truth = task.load_event_helper(ID)
    model_output = model(tf.expand_dims(original, axis=0))
    
    # for f in fillers:
    #     f.fill(tf_to_numpy(tf.reshape(original,[116,12])),tf_to_numpy(tf.reshape(truth,[116,12])),tf_to_numpy(tf.reshape(model_output,[116,12])))
    if model_output > 0.5:
        tf.print(original,summarize=-1)

        file = ROOT.TFile("my_generator_t%i_%i.root" % (ID[0],ID[1])) if ID[0] < 4 else ROOT.TFile("my_generator_t%i_%i_signal_like.root" % (ID[0]-3,ID[1]))
        tree = file.Get('hit_tree')
        tree.GetEntry(ID.numpy()[2])
        print("track split",tree.track_split)
        for i in range(tree.grid_layer.size()):
            print([tree.grid_layer[i],tree.grid_column[i]])

        input("")



def compare_models(ID,models):
    '''
    
    '''
    original,truth = task.load_event_helper(ID)
    my_print(tf_to_numpy(tf.reshape(original,[116,12])))
    
    for name,model in models.items():
        print(f"{name}")
        model_output = model(original)
        tf.print(original)
        print()
        # input("")




if __name__ == "__main__":
    # model = keras.models.load_model("../matteo_small_latetn_space_huge_run/model",compile=False)
    # model = keras.models.load_model("../with_skip_connection_latent_space_operations/model",compile=False)
    # model = keras.models.load_model("../VAE_architecture2_1/model")
    # model = keras.models.load_model("../GAN_tests/generator")
    # model = keras.models.load_model("../matteo_with_skip_connection2/model",compile=False)
    # model = keras.models.load_model("../clustering_matteo_with_skip_connection2/model",compile=False)

    # model = matteo.with_skip_cbam(None)
    # model.load_weights("../../attention/attention_final/model/variables/variables")

    # model = keras.models.load_model("../matteo_dummy/model",compile=False)
    # model = keras.models.load_model("../matteo_gan/generator",compile=False)

    # model = keras.models.load_model("../clustering_matteo_with_skip_connection/model",compile=False)




    # model = matteo.with_skip_cbam(None)
    # model.load_weights("../../attention/with_skip_latent_attention_run_2/model/variables/variables")

    # model = keras.models.load_model("../cbam_gan_finetuning_without_dis_pretraining/generator",compile=False)

    # model = keras.models.load_model("../../attention/bahdanau/model")


    # model = keras.models.load_model("../gan_not_access_thruth/generator",compile=False)

    # model = keras.models.load_model("/sps/nemo/scratch/amendl/AI/my_lib/optimizing_cbam/jobs_layer_normalization_sigmoid/0_16_0_16_16/model",compile=False)


    # draw_event(27,model)
    # draw_event(28,model)
    # draw_event(15,model)
    # draw_event(18,model)
    # draw_event(19,model)
    # draw_event(27,model)
    # draw_event(28,model)
    # exit(0)



    # finder  = ThresholdFinder()
    # metrics = SegmentationMetrics(10,0.5,"cbam_gan")

    model = architecture_with_normalization()
    model.load_weights("model/variables/variables")


    name = "final"
    print(model)
    for i in range(5000): 
        id = [3,9,i]
        # if i % 100 == 0:
        print(id)
        sys.stdout.flush()
        analyse_event(tf.constant(id),model,None)
    
    # metrics[0].draw_metrics("")[0].SaveAs(f"{name}_e.pdf")
    # metrics[0].draw_metrics("")[1].SaveAs(f"{name}_c.pdf")
    # metrics[1].draw_metrics("").SaveAs(f"{name}.pdf")
    # model = keras.models.load_model("../clustering_matteo_with_skip_connection2/model")

    # events = 5000


    # name = "final1"
    # metrics1 = [EMetric(0.5,name+"_e"),SegmentationMetrics(10,0.5,name)]
    # print(model)
    # for i in range(events): 
    #     if i % 100 == 0:
    #         print(f'2,9,{i}')
    #         sys.stdout.flush()
    #     analyse_event(tf.constant([2,9,i]),model,metrics1)
    # model = keras.models.load_model("/sps/nemo/scratch/amendl/AI/my_lib/optimizing_cbam/jobs_layer_normalization_sigmoid/0_16_0_16_16/model",compile=False)
    # name = "final2"
    # metrics2 = [EMetric(0.5,name+"_e"),SegmentationMetrics(10,0.5,name)]
    # print(model)
    # for i in range(events): 
    #     if i % 100 == 0:
    #         print(f'2,9,{i}')
    #         sys.stdout.flush()
    #     analyse_event(tf.constant([2,9,i]),model,metrics2)

     

    # canvas = ROOT.TCanvas()
    # metrics1[0].th1.Scale(1./ metrics1[0].th1.GetEntries())
    # a = metrics1[0].th1.GetCumulative(False)
    # a.SetStats(False)
    # a.SetLineColor(ROOT.kRed)
    # a.SetTitle("Cumulative;Not associated hits in event / %;")
    # a.Draw("HIST")
    # metrics2[0].th1.Scale(1./ metrics2[0].th1.GetEntries())
    # a = metrics2[0].th1.GetCumulative(False)
    # a.SetStats(False)
    # a.SetLineColor(ROOT.kBlue)
    # a.SetTitle("Cumulative;Not associated hits in event / %;")
    # a.Draw("HISTSAME")
    # canvas.SetLogy()
    # canvas.SaveAs("FINAL.pdf")

    

    #     if i % 10 == 0:
    #         print(i,flush=True)
    #         sys.stdout.flush()
    # metrics.plot_less_than_expected("less_than_expected.pdf")
    # metrics.plot_more_than_ecpected("more_than_expected.pdf")


    #     if i % 10 == 0:
    #         print(i,flush=True)

    # print(0.005+0.01*np.argmax(finder.histo))
    # plt.plot(np.linspace(0.005,1.-0.005,num=100),finder.histo)
    # plt.axvline(0.005+0.01*float(np.argmax(finder.histo)),color="red",linestyle="dashed")
    # plt.xlabel("threshold")
    # plt.ylabel("score (the bigger the better)")
    # plt.savefig("find.pdf")    
