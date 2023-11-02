#! /sps/nemo/scratch/amendl/AI/virtual_env_python391/bin/python
import tensorrt
import tensorflow as tf
from tensorflow import keras
import numpy as np
import sys

import ROOT
import matplotlib.pyplot as plt


from keras import backend as K

import math

import sys


TRACKS = 2



import tensorrt
from sklearn.metrics import confusion_matrix

import numpy as np

import tensorflow as tf
from tensorflow import keras
from keras import layers # Dense, Droupout, Softmax, BatchNormalization, Conv2D
from keras.callbacks import EarlyStopping
import tensorflow.keras.backend as K

import matplotlib.pyplot as plt

import sys
import datetime
import os 



def import_arbitrary_module(module_name,path):
    import importlib.util
    import sys

    spec = importlib.util.spec_from_file_location(module_name,path)
    imported_module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = imported_module
    spec.loader.exec_module(imported_module)

    return imported_module

task  = import_arbitrary_module("task","/sps/nemo/scratch/amendl/AI/my_lib/optimizing_cbam/scripts/dataset.py")
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


# def my_print(tensor):
#     # print(tensor.shape)
#     for i in range(116):
#         print(chr(9608),end="")
#     print()
#     for i in range(tensor.shape[0]):
#         print(chr(9608),end="")
#         for j in range(tensor.shape[1]):
#             if tensor[i,j]>0.545:
#                 print(chr(0x2299),end="")
#             else:
#                 print(" ",end="")
#         print(chr(9608))
#     for i in range(116):
#         print(chr(9608),end="")
    

# class ThresholdFinder:
#     '''
    
#     '''
#     def __init__(self):
#         self.histo = np.zeros((100))
#     def fill(self, original,truth,model_output):
#         for i in range(100):
#             threshold = 0.005 + float(i)*0.01
#             for j in range(model_output.shape[0]):
#                 for k in range(model_output.shape[1]):
#                     if original[j,k]>0.5:
#                         if (model_output[j,k] > threshold) == (truth[j,k]>0.5):
#                             self.histo[i]+=1
#                         else:
#                             self.histo[i]-=1




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

def cbam_optimization_layer_normalization(max_number_of_tracks):

    img_input = keras.layers.Input(shape=(116, 12))
    a = keras.layers.Reshape((116, 12,1))(img_input)
    conv1 = keras.layers.Conv2D(32, (3, 3), activation='relu', padding='same')(a)
    conv1 = keras.layers.Dropout(0.2)(conv1)
    conv1 = keras.layers.Conv2D(32, (3, 3), activation='relu', padding='same')(conv1)
    pool1 = keras.layers.MaxPooling2D((2, 2))(push_attention_block("P1",conv1,True))

    conv2 = keras.layers.Conv2D(64, (3, 3), activation='relu', padding='same')(pool1)
    conv2 = keras.layers.Dropout(0.2)(conv2)
    conv2 = keras.layers.Conv2D(64, (3, 3), activation='relu', padding='same')(conv2)
    pool2 = keras.layers.MaxPooling2D((2, 2), padding='same')(push_attention_block("P2",conv2,True))

    conv3 = keras.layers.Conv2D(128, (3, 3), activation='relu', padding='same')(pool2)
    conv3 = keras.layers.Dropout(0.2)(conv3)
    conv3 = keras.layers.Conv2D(128, (3, 3), activation='relu', padding='same')(conv3)


    up1 = keras.layers.concatenate([keras.layers.UpSampling2D((2, 2))(push_attention_block("P3",conv3,True)), conv2], axis=-1)
    conv4 = keras.layers.Conv2D(64, (3, 3), activation='relu', padding='same')(up1)
    conv4 = keras.layers.Dropout(0.2)(push_attention_block("P4",conv4,True))
    conv4 = keras.layers.Conv2D(64, (3, 3), activation='relu', padding='same')(conv4)

    up2 = keras.layers.concatenate([keras.layers.UpSampling2D((2, 2))(conv4), conv1], axis=-1)
    conv5 = keras.layers.Conv2D(32, (3, 3), activation='relu', padding='same')(up2)
    conv5 = keras.layers.Dropout(0.2)(push_attention_block("P5",conv5,True))
    conv5 = keras.layers.Conv2D(32, (3, 3), activation='relu', padding='same')(conv5)

    out = keras.layers.Conv2D(max_number_of_tracks + 1, 3, padding='same',activation='sigmoid')(keras.layers.UpSampling2D(1)(conv5))
    out = keras.layers.Softmax()(out)

    model = keras.models.Model(img_input, out)

    return model


class SegmentationMetrics:
    '''
            
    '''
    def __init__(self,histo_sizes,threshold,name):
        self.threshold = threshold
        self.histo_sizes = histo_sizes
        self.randomly_added     = []
        self.th2 = ROOT.TH2D(name,name,20,-0.01,100.01,20,-0.01,100.01)
        self.th1 = ROOT.TH1D(name+"_distance",name,20,-0.01,150.1)
        self.name = name
    def fill(self, original,truth,model_output):
        n_of_preserved_truth  = 0
        n_of_removed_truth    = 0
        less_than_expected    = 0
        more_than_expected    = 0
        # print(model_output.shape[0],model_output.shape[1])
        # print(original.shape[0],original.shape[1])
        # print(truth.shape[0],truth.shape[1])

        for j in range(model_output.shape[0]):
            for k in range(model_output.shape[1]):
                if original[j,k] > 0.5:
                    # truth
                    if truth[j,k] > 0.5:
                        n_of_preserved_truth+=1
                    if truth[j,k] < 0.5:
                        n_of_removed_truth+=1
                    # model
                    if model_output[j,k] < self.threshold and truth[j,k] > 0.5:
                        less_than_expected+=1
                    if model_output[j,k] > self.threshold and truth[j,k] < 0.5:
                        more_than_expected+=1
        # fill
        # print(n_of_preserved_truth,n_of_removed_truth)
        # print(more_than_expected)
        self.th2.Fill(float(less_than_expected)/float(n_of_preserved_truth)*100.,float(more_than_expected)/float(n_of_removed_truth)*100.)
        self.th1.Fill(100*math.sqrt(float(less_than_expected)/float(n_of_preserved_truth)*float(less_than_expected)/float(n_of_preserved_truth) + float(more_than_expected)/float(n_of_removed_truth)*float(more_than_expected)/float(n_of_removed_truth)))

    def draw_metrics(self,text,f):
        canvas = ROOT.TCanvas()
        canvas.Divide(2,2)
        canvas.cd(2)
        self.th2.SetTitle("Metric C: Correlation between A, B;Percent of right track removed;Percent of left track left")
        self.th2.SetStats(False)
        self.th2.Draw("COLZ")
        canvas.cd(1).SetLogy()
        pX = self.th2.ProjectionX()
        pX.SetTitle("Metric A: Percent of left track left;Percent of left track left;N")
        f.write(f"{pX.GetMean(1)};{pX.GetStdDev(1)}\n")
        pX.Draw()
        canvas.cd(4).SetLogy()
        pY = self.th2.ProjectionY()
        pY.SetTitle("Metric B: Percent of right track removed;Percent of right track removed;N")
        f.write(f"{pY.GetMean(1)};{pY.GetStdDev(1)}\n")
        pY.Draw()
        canvas.cd(3).SetLogy()
        self.th1.SetTitle("Total distance;Total distance percent;N")
        f.write(f"{self.th1.GetMean(1)};{self.th1.GetStdDev(1)}\n")
        self.th1.Draw()
        # l = ROOT.TLatex()
        # l.SetTextAlign(20)
        # l.DrawLatexNDC(0.5, 0.5, text)
        return canvas
            
    def plot_randomly_added(self,params):
        raise NotImplementedError

class EMetric:
    '''
    
    '''
    def __init__(self,threshold,name):
        self.threshold = threshold
        self.name      = name
        self.th1       = ROOT.TH1D(name,name,20,-0.01,100.01)
    
    def fill(self, original,truth,model_output):
        n_in_reconstructed = 0
        n_correct          = 0
        for j in range(model_output.shape[0]):
            for k in range(model_output.shape[1]):
                if model_output[j,k] > self.threshold:
                    n_in_reconstructed += 1
                    if truth[j,k] > 0.5:
                        n_correct += 1
        if n_in_reconstructed != 0:
            self.th1.Fill(float(n_in_reconstructed-n_correct)/float(n_in_reconstructed)*100.)
    
    def draw_metrics(self,text,f):
        canvas = ROOT.TCanvas()
        self.th1.SetTitle("Metric E: Percent of hits in clustered event not associated with right track;Percent of hits in clustered event not associated with right track;N")
        f.write(f"{self.th1.GetMean(1)},{self.th1.GetStdDev(1)};")
        self.th1.Draw()
        canvas.SetLogy()
        canvas2 = ROOT.TCanvas()
        self.th1.GetCumulative(False).DrawNormalized()
        canvas2.SetLogy()
        canvas2.Draw()
        return canvas,canvas2


def analyse_event(ID,model,fillers):
    original,truth = task.load_event_helper(ID)
    model_output = model(tf.reshape(original,[1,116,12]))
    # for f in fillers:
    #     for i in range(TRACKS):
    #         f[i].fill(tf_to_numpy(tf.reshape(original,[116,12])),tf_to_numpy(tf.reshape(tf.constant(truth.numpy()),[116,12])),tf_to_numpy(tf.reshape(tf.constant(model_output.numpy()[:,:,i]),[116,12])))


    my_print(tf_to_numpy(tf.reshape(original,[116,12])))
    print()
    tf.print(tf.reshape(tf.constant(np.argmax(model_output.numpy(),axis=3)),[116,12]),summarize=-1)
    # tf.print(model_output,summarize=-1)

    # print(tf.shape(model_output))
    print()
    input("")



if __name__ == "__main__":
    print("Loading model")
    sys.stdout.flush()
    model = cbam_optimization_layer_normalization(TRACKS)
    model.load_weights("model/variables/variables")
    print("Initializing metrics")
    sys.stdout.flush()
    name = sys.argv[1]
    metrics = [(EMetric(0.5,name+"_e"),SegmentationMetrics(10,0.5,name)) for _ in range(TRACKS)]
    print(model)
    print("Starting loop with 10000 elements")
    sys.stdout.flush()
    for i in range(10000): 
        if i % 100 == 0:
            print(f'2,9,{i}')
            sys.stdout.flush()
        analyse_event(tf.constant([2,9,i]),model,metrics)
    with open("data.txt",'w') as f:
        metrics[0].draw_metrics("",f)[0].SaveAs(f"{name}_e.pdf")
        metrics[0].draw_metrics("",f)[1].SaveAs(f"{name}_c.pdf")
        metrics[1].draw_metrics("",f).SaveAs(f"{name}.pdf")


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
